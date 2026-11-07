package io.github.a775828b.xtiergo

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * The launcher's state: the cards, the chosen card and its option values (saved per card), its files, the
 * memory estimate (what does not fit is greyed with the reason) and the running server's status, polled from
 * http://127.0.0.1:<port>/xtg/status (the server lives in the :engine process).
 */
class LauncherModel(private val app: Context) {
    val cards: List<Card> = Card.loadAll(app)
    val store = ModelStore(app)
    private val prefs = app.getSharedPreferences("launcher", Context.MODE_PRIVATE)

    var selected by mutableStateOf(prefs.getString("selected", null) ?: cards.firstOrNull()?.id.orEmpty())
        private set
    var values by mutableStateOf<Map<String, String>>(emptyMap())
        private set
    var files by mutableStateOf<List<ModelStore.Status>>(emptyList())
        private set
    var status by mutableStateOf<JSONObject?>(null)
        private set
    var device by mutableStateOf(DeviceInfo.read(app))
        private set
    var estimate by mutableStateOf<JSONObject?>(null)
        private set
    // option key -> value -> why it does not fit (from the memory estimate)
    var tooBig by mutableStateOf<Map<String, Map<String, String>>>(emptyMap())
        private set
    var message by mutableStateOf("")

    init {
        if (cards.none { it.id == selected }) selected = cards.firstOrNull()?.id.orEmpty()
        if (selected.isNotEmpty()) values = card().resolve(Card.values(prefs.getString("values:$selected", null)))
    }

    fun card(): Card = cards.first { it.id == selected }

    fun select(id: String) {
        selected = id
        prefs.edit().putString("selected", id).apply()
        values = card().resolve(Card.values(prefs.getString("values:$id", null)))
    }

    fun set(key: String, value: String) {
        values = values + (key to value)
        prefs.edit().putString("values:$selected", Card.valuesJson(values)).apply()
    }

    fun reset() {
        values = card().defaults()
        prefs.edit().remove("values:$selected").apply()
    }

    fun start() {
        val c = card()
        if (!store.ready(c)) {
            message = "模型文件还不全：先下载或用 adb push 到 ${store.dir(c).path}"
            return
        }
        message = ""
        EngineService.start(app, c.launchConfig(store.dir(c).path, values))
    }

    fun stop() = EngineService.stop(app)

    // the download of the chosen card (DownloadService), refreshed by poll()
    var downloading by mutableStateOf(false)
        private set
    var downloadProgress by mutableStateOf<Downloader.Progress?>(null)
        private set
    var downloadResult by mutableStateOf("")
        private set

    fun download(mirror: String) {
        DownloadService.start(app, card(), mirror)
        message = "开始下载：${mirror.removePrefix("https://")}（直连，域名经 223.5.5.5 解析）"
    }

    fun cancelDownload() = DownloadService.cancel(app)

    suspend fun verify() {
        message = "正在校验 sha256…"
        val bad = withContext(Dispatchers.IO) { store.verify(card()) }
        message = if (bad.isEmpty()) "校验通过" else "校验不通过：${bad.joinToString()}（删掉重新下载）"
    }

    /** Polls status, device and files every second; the estimate when the options or files change. */
    suspend fun poll() {
        var lastKey = ""
        while (true) {
            val c = card()
            status = fetchStatus(c.port)
            device = withContext(Dispatchers.IO) { DeviceInfo.read(app) }
            files = withContext(Dispatchers.IO) { store.status(c) }
            downloading = Downloads.running && Downloads.cardId == c.id
            downloadProgress = if (Downloads.cardId == c.id) Downloads.progress else null
            downloadResult = if (Downloads.cardId == c.id) Downloads.result else ""
            val ready = files.all { it.state == ModelStore.FileState.Ready }
            val key = "$selected|$values|$ready|${device.availBytes / (256L shl 20)}"
            if (key != lastKey) {
                lastKey = key
                if (ready) refreshEstimate(c) else {
                    estimate = null
                    tooBig = emptyMap()
                }
            }
            delay(1000)
        }
    }

    private suspend fun refreshEstimate(c: Card) = withContext(Dispatchers.IO) {
        val dir = store.dir(c).path
        estimate = runCatching { JSONObject(Native.estimate(dir, c.packOptions(values).toString())) }.getOrNull()
        // a server already running holds its memory: count it as available for the next launch
        val running = status?.optJSONObject("telemetry")?.optLong("rss_bytes", 0L) ?: 0L
        val budget = device.availBytes + running - (1536L shl 20)
        val out = mutableMapOf<String, Map<String, String>>()
        for (o in c.options.filter { it.target == "option" }) {
            val reasons = mutableMapOf<String, String>()
            for (ch in o.choices) {
                val e = runCatching {
                    JSONObject(Native.estimate(dir, c.packOptions(values + (o.key to ch.value)).toString()))
                }.getOrNull() ?: continue
                val need = e.optLong("resident_bytes", 0L)
                if (need > budget) reasons[ch.value] = "需要约 ${gib(need)}，可用 ${gib(budget.coerceAtLeast(0))}"
            }
            if (reasons.isNotEmpty()) out[o.key] = reasons
        }
        tooBig = out
    }

    private suspend fun fetchStatus(port: Int): JSONObject? = withContext(Dispatchers.IO) {
        try {
            val c = URL("http://127.0.0.1:$port/xtg/status").openConnection() as HttpURLConnection
            c.connectTimeout = 400
            c.readTimeout = 1500
            c.inputStream.bufferedReader().use { JSONObject(it.readText()) }
        } catch (e: Exception) {
            null
        }
    }
}
