package io.github.a775828b.xtiergo

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

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
    // the console's log (state changes, load phases, finished requests, benchmarks) and the decode rate per poll
    var events by mutableStateOf<List<String>>(emptyList())
        private set
    var spark by mutableStateOf<List<Float>>(emptyList())
        private set
    var benching by mutableStateOf(false)
        private set
    private var seenState = ""
    private var seenPhase = ""
    private var seenRequests = -1

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
        val config = c.launchConfig(store.dir(c).path, values)
        log("ENGAGE · ${c.code} · ${config.optJSONObject("options")} · ${config.optInt("threads")} 线程")
        EngineService.start(app, config)
    }

    fun stop() {
        log("ABORT")
        EngineService.stop(app)
    }

    fun log(text: String) {
        events = (events + "${SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())} $text").takeLast(60)
    }

    /** One fixed benchmark through the API: flush the prefix cache, a fixed prompt, temperature 0, thinking off. */
    suspend fun bench(b: Bench) {
        val c = card()
        benching = true
        log("测速 · ${b.label} · 开始")
        val line = try {
            withContext(Dispatchers.IO) {
                post(c.port, "/flush_cache", "{}")
                val body = JSONObject()
                    .put("model", c.served)
                    .put("messages", JSONArray().put(JSONObject().put("role", "user").put("content", benchPrompt(b.repeat))))
                    .put("max_tokens", b.maxTokens)
                    .put("temperature", 0)
                    .put("chat_template_kwargs", JSONObject().put("enable_thinking", false))
                val t0 = System.nanoTime()
                val r = JSONObject(post(c.port, "/v1/chat/completions", body.toString(), 3_600_000))
                val wall = (System.nanoTime() - t0) / 1e9
                val u = r.optJSONObject("usage") ?: JSONObject()
                val last = fetchStatus(c.port)?.optJSONObject("telemetry")?.optJSONObject("last") ?: JSONObject()
                "测速 · ${b.label} · ${u.optInt("prompt_tokens")} 提示 · ${u.optInt("completion_tokens")} 生成 · " +
                    "prefill %.1f · decode %.1f tok/s · 共 %.1f s".format(last.optDouble("prefill_tok_s", 0.0),
                        last.optDouble("decode_tok_s", 0.0), wall)
            }
        } catch (e: Exception) {
            "测速失败：${e.message}"
        }
        log(line)
        benching = false
    }

    suspend fun flushCache() {
        val c = card()
        val r = withContext(Dispatchers.IO) { runCatching { post(c.port, "/flush_cache", "{}") } }
        log(if (r.isSuccess) "前缀缓存已清" else "清缓存失败：${r.exceptionOrNull()?.message}")
    }

    val npuTools = NpuProbe.available(app)
    var npuChecking by mutableStateOf(false)
        private set

    /** NpuProbe from this process; every line goes to the log, the report to npu/selftest.txt. */
    suspend fun npuSelfTest() {
        if (npuChecking) return
        npuChecking = true
        log("NPU 自检 · 开始（${NpuProbe.dir(app).path}）")
        val lines = withContext(Dispatchers.IO) { runCatching { NpuProbe.run(app) } }
        lines.getOrNull()?.forEach { log("NPU · $it") } ?: log("NPU 自检失败：${lines.exceptionOrNull()?.message}")
        npuChecking = false
    }

    data class Bench(val label: String, val repeat: Int, val maxTokens: Int)

    companion object {
        val BENCHES = listOf(Bench("短 ~300 提示", 2, 32), Bench("长 ~2K 提示", 14, 64))

        // English and Chinese, about 150 tokens per repeat
        private const val BENCH_TEXT = "XTier Go runs a looped language model on this tablet: the launcher starts a local server, " +
            "other apps call its OpenAI compatible API on 127.0.0.1, and the engine keeps the KV cache in FP8 so that a long " +
            "agent session fits in memory. 平板上的推理服务要同时照顾速度、内存和发热：prefill 受算力限制，decode 受内存带宽限制，" +
            "长上下文时注意力读取的 KV 越来越多。请记住这些约束。"

        fun benchPrompt(repeat: Int): String =
            (1..repeat).joinToString("\n") { "[$it] $BENCH_TEXT" } + "\n\n请用三句话总结上面的内容。"
    }

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
            note(status)
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

    /** The log lines and the spark samples a new status brings. */
    private fun note(s: JSONObject?) {
        val state = uiState(s)
        if (state != seenState) {
            when (state) {
                "online" -> {
                    val d = s?.optJSONObject("model")
                    log("ONLINE · " + (d?.let { "${it.optInt("loops")} 循环 · ${it.optString("kv_mode")} · ${it.optString("kv_dtype")} · ${it.optString("kernels")}" } ?: ""))
                    spark = emptyList()
                }
                "failed" -> log("FAILED · ${s?.optString("error").orEmpty()}")
                "offline" -> if (seenState.isNotEmpty()) log("OFFLINE")
            }
            seenState = state
        }
        if (state == "loading") {
            val phase = s?.optString("phase").orEmpty()
            if (phase != seenPhase) log("加载 · ${s?.optString("phase_label").orEmpty()} · ${s?.optInt("progress") ?: 0}%")
            seenPhase = phase
        } else seenPhase = ""
        val t = s?.optJSONObject("telemetry")
        if (state != "online" || t == null) {
            seenRequests = -1
            return
        }
        val n = t.optInt("requests")
        if (seenRequests in 0 until n) t.optJSONObject("last")?.let { last ->
            log("请求 #$n · ${last.optInt("prompt_tokens")} 提示（命中 ${last.optInt("cached_tokens")}）· ${last.optInt("completion_tokens")} 生成 · " +
                "prefill %.1f · decode %.1f · %s".format(last.optDouble("prefill_tok_s", 0.0), last.optDouble("decode_tok_s", 0.0),
                    last.optString("finish")))
        }
        seenRequests = n
        spark = (spark + t.optDouble("decode_tok_s", 0.0).toFloat()).takeLast(90)
    }

    private fun post(port: Int, path: String, body: String, timeoutMs: Int = 5000): String {
        val c = URL("http://127.0.0.1:$port$path").openConnection() as HttpURLConnection
        c.requestMethod = "POST"
        c.doOutput = true
        c.connectTimeout = 1000
        c.readTimeout = timeoutMs
        c.setRequestProperty("Content-Type", "application/json")
        c.outputStream.use { it.write(body.toByteArray()) }
        val code = c.responseCode
        val text = (if (code in 200..299) c.inputStream else c.errorStream)?.bufferedReader()?.use { it.readText() }.orEmpty()
        check(code in 200..299) { "HTTP $code ${text.take(200)}" }
        return text
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
