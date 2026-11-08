package io.github.a775828b.xtiergo

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import org.json.JSONObject

// The launcher in the XTier console's layout (xtier/console/ui.py): a model rail, the launch parameters of the
// chosen model (defaults = the card's measured configuration, changed ones marked) and the telemetry HUD that
// follows the loading phases and, once online, the server's decode rate. Three columns on a wide screen, the rail
// on top below 1080 dp, one scrolling column on a phone.

private const val SUBTITLE = "本机推理 · 整步跑在高通 NPU（Hexagon V79，ExecuTorch QNN 静态图），Adreno GPU / CPU 备用 · 127.0.0.1 上的 OpenAI 兼容接口，给 DSH Go X 用"
private const val FOOT = "默认参数 = configs/*/card.json 里的实测配置，改动过的项会标出。一次只跑一个模型；服务在后台进程里，关掉启动台不影响它。"

private val PHASES = listOf("files" to "文件", "tokenizer" to "分词器", "weights" to "权重", "kv" to "KV 缓存",
    "warmup" to "预热", "online" to "ONLINE")

@Composable
fun Launcher(m: LauncherModel, multiWindow: Boolean) {
    LaunchedEffect(Unit) { m.poll() }
    // split screen / a floating window: the rate board next to the app that calls the server, the full console a tap away
    var board by remember(multiWindow) { mutableStateOf(multiWindow) }
    if (board && m.cards.isNotEmpty()) {
        RateBoard(m) { board = false }
        return
    }
    Console(Modifier.fillMaxSize().background(Xt.bottom).safeDrawingPadding().padding(8.dp)) {
        if (m.cards.isEmpty()) {
            Text("没有模型卡片（assets/*/card.json）", color = Xt.r, modifier = Modifier.padding(24.dp))
            return@Console
        }
        Column(Modifier.fillMaxSize()) {
            Header(m, if (multiWindow) ({ board = true }) else null)
            BoxWithConstraints(Modifier.weight(1f).fillMaxWidth()) {
                when {
                    maxWidth >= 1080.dp -> Row(Modifier.fillMaxSize().padding(16.dp), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                        Rail(m, Modifier.width(256.dp).fillMaxHeight(), column = true)
                        Form(m, Modifier.weight(1.1f).fillMaxHeight(), scroll = true)
                        Telemetry(m, Modifier.weight(1f).fillMaxHeight(), scroll = true)
                    }
                    maxWidth >= 720.dp -> Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                        Rail(m, Modifier.fillMaxWidth(), column = false)
                        Row(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                            Form(m, Modifier.weight(1.1f).fillMaxHeight(), scroll = true)
                            Telemetry(m, Modifier.weight(1f).fillMaxHeight(), scroll = true)
                        }
                    }
                    else -> Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Rail(m, Modifier.fillMaxWidth(), column = false)
                        Form(m, Modifier.fillMaxWidth(), scroll = false)
                        Telemetry(m, Modifier.fillMaxWidth(), scroll = false)
                    }
                }
            }
            Dim(FOOT, 11.5.sp, color = Xt.faint, lines = 2)
            Spacer(Modifier.height(4.dp))
        }
    }
}

@Composable
private fun Header(m: LauncherModel, onRates: (() -> Unit)?) {
    val s = m.status
    val state = uiState(s)
    val c = m.card()
    val sub = when (state) {
        "online" -> "${c.name} · :${c.port}"
        "loading" -> "${s?.optString("phase_label").orEmpty()} · ${s?.optInt("progress") ?: 0}%"
        "failed" -> s?.optString("error").orEmpty()
        else -> "待命"
    }
    BoxWithConstraints(Modifier.fillMaxWidth().drawBehind {
        drawLine(Xt.line, Offset(0f, size.height), Offset(size.width, size.height), 1.dp.toPx())
    }) {
        val narrow = maxWidth < 640.dp
        val brand = @Composable {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                XtLogo()
                Column {
                    Row(verticalAlignment = Alignment.Bottom) {
                        GradientTitle("XTIER GO")
                        Spacer(Modifier.width(10.dp))
                        Text("ON-DEVICE CONSOLE", color = Xt.d, fontSize = 11.sp, letterSpacing = .32.em, maxLines = 1)
                    }
                    Dim(SUBTITLE, 12.5.sp, lines = 2)
                }
            }
        }
        if (narrow) Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            brand()
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                StatePill(state, sub, Modifier.weight(1f))
                if (onRates != null) XtButton("速率表", small = true, onClick = onRates)
            }
        } else Row(Modifier.padding(start = 22.dp, end = 22.dp, top = 16.dp, bottom = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) { brand() }
            Spacer(Modifier.width(16.dp))
            StatePill(state, sub, Modifier.widthIn(max = 420.dp))
            if (onRates != null) {
                Spacer(Modifier.width(10.dp))
                XtButton("速率表", small = true, onClick = onRates)
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun Rail(m: LauncherModel, modifier: Modifier, column: Boolean) {
    val s = m.status
    val live = uiState(s)
    XtPanel("MODELS · 模型", modifier, scroll = column) {
        Dim("README 实测 tok/s · 文本：短提示 · AGENT：编码代理会话", 10.sp)
        Spacer(Modifier.height(9.dp))
        // cards share the port: the server's model directory says which card it runs
        val dir = s?.optJSONObject("config")?.optString("model_dir").orEmpty()
        val tab = @Composable { c: Card, mod: Modifier ->
            val mine = s != null && s.optInt("port") == c.port && (dir.isEmpty() || dir.endsWith("/" + c.dir))
            val running = if (mine && (live == "online" || live == "loading")) live else ""
            ModelTab(c, c.id == m.selected, running, mod) { m.select(c.id) }
        }
        if (column) Column(verticalArrangement = Arrangement.spacedBy(8.dp)) { for (c in m.cards) tab(c, Modifier.fillMaxWidth()) }
        else FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            for (c in m.cards) tab(c, Modifier.width(240.dp))
        }
        Spacer(Modifier.height(12.dp))
        val d = m.device
        Dim("${d.soc} · ${d.cpuFeatures.ifEmpty { "NEON" }}", 11.sp, mono = true)
        Dim("内存 可用 ${gib(d.availBytes)} / ${gib(d.totalBytes)}", 11.sp)
    }
}

@Composable
private fun Form(m: LauncherModel, modifier: Modifier, scroll: Boolean) {
    val c = m.card()
    val state = uiState(m.status)
    val busy = state == "loading" || state == "online"
    XtPanel("LAUNCH PARAMETERS · 启动参数", modifier, scroll) {
        Column(Modifier.fillMaxWidth().drawBehind {
            drawLine(Xt.c.copy(alpha = .16f), Offset(0f, size.height), Offset(size.width, size.height), 1.dp.toPx(),
                pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(6f, 6f)))
        }.padding(bottom = 10.dp)) {
            Text(c.name, color = Xt.bright, fontSize = 15.sp, fontWeight = FontWeight.Bold)
            Dim(c.arch, 12.sp)
            Dim("${c.hardware.ifEmpty { m.device.soc }} · :${c.port}", 11.sp, color = Xt.c, mono = true)
        }
        Spacer(Modifier.height(12.dp))
        FilesBox(m, c)
        Spacer(Modifier.height(12.dp))
        Options(m, c)
        val notes = c.activeNotes(m.values)
        if (notes.isNotEmpty()) Spacer(Modifier.height(10.dp))
        for (n in notes) Dim((if (n.warn) "⚠ " else "· ") + n.text, 12.sp, color = if (n.warn) Xt.a else Xt.d)
        m.estimate?.let { e ->
            Spacer(Modifier.height(6.dp))
            Dim("预计常驻 ${gib(e.optLong("resident_bytes"))}（权重 ${gib(e.optLong("weights_bytes") - e.optLong("embedding_bytes"))}" +
                " · KV 满上下文 ${gib(e.optLong("kv_full_bytes") + e.optLong("kv_window_bytes"))}" +
                " · 快照 ${gib(e.optLong("snapshot_bytes"))}），可用 ${gib(m.device.availBytes)}", 12.sp)
        }
        Spacer(Modifier.height(14.dp))
        @OptIn(ExperimentalLayoutApi::class)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            EngageButton(if (state == "online") "重启 · RESTART" else "启动 · ENGAGE", enabled = !busy || state == "online") { m.start() }
            XtButton("停止 · ABORT", enabled = busy, danger = true) { m.stop() }
            XtButton("恢复实测默认") { m.reset() }
        }
        if (m.message.isNotEmpty()) {
            Spacer(Modifier.height(10.dp))
            Dim(m.message, 12.5.sp, color = if (m.message.startsWith("校验通过")) Xt.g else Xt.a)
        }
        if (state == "failed") {
            Spacer(Modifier.height(6.dp))
            Dim(m.status?.optString("error").orEmpty(), 12.5.sp, color = Xt.r)
        }
    }
}

@Composable
private fun Options(m: LauncherModel, c: Card) {
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        val cols = if (maxWidth >= 460.dp) 2 else 1
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            for (row in c.options.chunked(cols)) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    for (o in row) {
                        val v = m.values[o.key] ?: o.default
                        val off = c.isOff(o, m.values)
                        val reasons = o.unavailable + (m.tooBig[o.key] ?: emptyMap())
                        if (o.switch && o.choices.size == 2) {
                            val on = v == o.choices[0].value
                            SwitchField(o.label, on, o.choices[if (on) 0 else 1].label, v != o.default, off, o.hint, Modifier.weight(1f)) {
                                m.set(o.key, o.choices[if (on) 1 else 0].value)
                            }
                        } else SelectField(
                            o.label, o.choices.map { Pick(it.value, it.label, reasons[it.value]) }, v, v != o.default, off, o.hint,
                            Modifier.weight(1f),
                        ) { m.set(o.key, it) }
                    }
                    if (row.size < cols) Spacer(Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun FilesBox(m: LauncherModel, c: Card) {
    val scope = rememberCoroutineScope()
    val ready = m.files.isNotEmpty() && m.files.all { it.state == ModelStore.FileState.Ready }
    val shape = RoundedCornerShape(10.dp)
    Column(Modifier.fillMaxWidth().clip(shape).background(Xt.field).border(1.dp, if (ready) Xt.line else Xt.a.copy(alpha = .4f), shape)
        .padding(10.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("MODEL FILES · 模型文件", color = Xt.c, fontSize = 10.5.sp, letterSpacing = .2.em, fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f))
            Text(if (ready) "READY" else "MISSING", color = if (ready) Xt.g else Xt.a, fontSize = 10.sp, letterSpacing = .16.em,
                fontFamily = Xt.mono)
        }
        Dim("${c.repo} @ ${c.revision.take(8)}", 11.sp, mono = true, lines = 1)
        Dim(m.store.dir(c).path, 10.5.sp, mono = true, lines = 2)
        for (f in m.files) {
            val (label, color) = when (f.state) {
                ModelStore.FileState.Ready -> "就绪" to Xt.g
                ModelStore.FileState.Unverified -> "待校验" to Xt.a
                ModelStore.FileState.Downloading -> "下载中 ${gib(f.bytes)} / ${gib(f.size)}" to Xt.c
                ModelStore.FileState.Partial -> "不完整 ${gib(f.bytes)} / ${gib(f.size)}" to Xt.a
                ModelStore.FileState.Bad -> "大小或校验不对" to Xt.r
                ModelStore.FileState.Missing -> (if (f.local) "缺 · adb push" else "缺") to Xt.a
            }
            Row {
                Dim(f.name, 12.sp, mono = true, lines = 1)
                Spacer(Modifier.weight(1f))
                Text(label, color = color, fontSize = 12.sp)
            }
        }
        val p = m.downloadProgress
        when {
            m.downloading -> {
                XtBar(if (p == null) "准备下载…" else "${gib(p.done)} / ${gib(p.total)} · %.1f MB/s · %s".format(p.bytesPerSec / 1e6, p.file),
                    if (p != null && p.total > 0) p.done.toFloat() / p.total else 0f)
                if (p != null && p.note.isNotEmpty()) Dim(p.note, 11.sp)
                XtButton("停止下载", small = true, danger = true) { m.cancelDownload() }
            }
            !ready -> {
                val fetch = m.files.any { !it.local && it.state != ModelStore.FileState.Ready }
                if (fetch) Dim("直连下载：独立域名解析，不经系统 HTTP 代理（VPN 仍可能影响连接），中断后可续传", 11.sp)
                if (m.files.any { it.local && it.state != ModelStore.FileState.Ready })
                    Dim("标 adb push 的文件没有下载源：在服务器上导出（docs/npu.md）后 adb push 到上面的目录", 11.sp)
                @OptIn(ExperimentalLayoutApi::class)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (fetch) for (mirror in c.mirrors) XtButton("下载 · ${mirror.removePrefix("https://")}", small = true) { m.download(mirror) }
                    if (m.files.any { it.state == ModelStore.FileState.Unverified })
                        XtButton("校验已有文件", small = true) { scope.launch { m.verify() } }
                }
            }
            else -> XtButton("校验 sha256", small = true) { scope.launch { m.verify() } }
        }
        if (m.downloadResult.isNotEmpty() && !m.downloading)
            Dim(m.downloadResult, 12.sp, color = if (m.downloadResult.startsWith("下载完成")) Xt.g else Xt.a)
    }
}

@Composable
private fun Telemetry(m: LauncherModel, modifier: Modifier, scroll: Boolean) {
    val s: JSONObject? = m.status
    val state = uiState(s)
    val c = m.card()
    val t = s?.optJSONObject("telemetry") ?: JSONObject()
    val model = s?.optJSONObject("model")
    val scope = rememberCoroutineScope()
    XtPanel("TELEMETRY · 运行状态", modifier, scroll) {
        val progress = when (state) {
            "online" -> 1f
            "loading" -> (s?.optInt("progress") ?: 0) / 100f
            else -> 0f
        }
        val decode = t.optDouble("decode_tok_s", 0.0)
        val (big, sub) = when (state) {
            "online" -> (if (decode > 0) "%.1f".format(decode) else "%.1f".format(t.optDouble("last_decode_tok_s", 0.0))) to
                (if (decode > 0) "DECODE TOK/S" else "LAST DECODE")
            "loading" -> "${s?.optInt("progress") ?: 0}%" to (s?.optString("phase_label")?.ifEmpty { null } ?: "LOADING")
            "failed" -> "ERR" to "FAILED"
            else -> "--" to "STANDBY"
        }
        val hud = @Composable { inner: Modifier ->
            Column(inner, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Phases(PHASES, if (state == "online") "online" else s?.optString("phase"), state == "online")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    val on = state == "online"
                    Meter("DECODE", if (on) "%.1f".format(t.optDouble("last_decode_tok_s", 0.0)) else "--", "tok/s", Modifier.weight(1f))
                    Meter("PREFILL", if (on) "%.1f".format(t.optDouble("last_prefill_tok_s", 0.0)) else "--", "tok/s", Modifier.weight(1f))
                    Meter("运行请求", if (on) "${if (t.optBoolean("running")) 1 else 0} / ${t.optInt("queued")}" else "--", "", Modifier.weight(1f))
                }
                Spark(m.spark)
            }
        }
        BoxWithConstraints(Modifier.fillMaxWidth()) {
            if (maxWidth >= 420.dp) Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                Ring(progress, state == "online", big, sub)
                hud(Modifier.weight(1f))
            } else Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Ring(progress, state == "online", big, sub)
                hud(Modifier.fillMaxWidth())
            }
        }
        Spacer(Modifier.height(10.dp))
        val d = m.device
        val rss = t.optLong("rss_bytes", 0L)
        XtBar("内存 · 进程 ${if (state == "online" || state == "loading") gib(rss) else "--"} · 可用 ${gib(d.availBytes)} / ${gib(d.totalBytes)}",
            if (d.totalBytes > 0) 1f - d.availBytes.toFloat() / d.totalBytes else 0f, warn = d.lowMemory)
        Spacer(Modifier.height(8.dp))
        val head = if (d.thermalHeadroom.isNaN()) 0f else d.thermalHeadroom
        XtBar("热 · ${DeviceInfo.thermalLabel(d.thermalStatus)}" + (if (!d.thermalHeadroom.isNaN()) " · 余量 %.2f".format(d.thermalHeadroom) else "") +
            " · 电池 %.1f °C · %d%%".format(d.batteryTempC, d.batteryPercent) + (if (d.charging) " · 充电中" else "") +
            (if (!d.batteryWatts.isNaN()) " · %.1f W".format(d.batteryWatts) else ""), head, warn = head >= .85f)
        Spacer(Modifier.height(12.dp))
        @OptIn(ExperimentalLayoutApi::class)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Dim("接口", 12.sp)
            CodeChip("http://127.0.0.1:${c.port}/v1")
            Dim("模型名", 12.sp)
            CodeChip(c.served)
        }
        if (model != null) {
            Spacer(Modifier.height(8.dp))
            val ctx = model.optLong("context")
            val ctxLabel = if (ctx % 1024 == 0L) "${ctx / 1024}K" else "%.1fK".format(ctx / 1024.0)
            if (model.optString("backend") == "npu")
                Dim("NPU · ${model.optInt("loops")} 次循环 · 上下文 $ctxLabel · prefill ${model.optInt("prefill_chunk")} 行 / decode " +
                    "${model.optInt("decode_rows")} 行 · 8 位 KV（${gib(model.optLong("kv_bytes"))}）· ${model.optString("pte")}", 11.5.sp, mono = true)
            else
                Dim("${model.optInt("loops")} 次循环 · ${model.optString("kv_mode")} · ${model.optString("kv_dtype")} KV · 上下文 $ctxLabel · " +
                    "${model.optString("kernels")} · ${model.optInt("threads")} 线程" +
                    model.optString("gpu_device").let { if (it.isNotEmpty()) " · $it" else "" }, 11.5.sp, mono = true)
        }
        t.optJSONObject("last")?.let { last ->
            Dim("上一请求 ${last.optInt("prompt_tokens")} 提示（命中 ${last.optInt("cached_tokens")}）· ${last.optInt("completion_tokens")} 生成 · " +
                "prefill %.1f · decode %.1f tok/s · %s".format(last.optDouble("prefill_tok_s", 0.0), last.optDouble("decode_tok_s", 0.0),
                    last.optString("finish")), 11.5.sp, mono = true)
        }
        Spacer(Modifier.height(10.dp))
        LogBox(m.events)
        Spacer(Modifier.height(12.dp))
        DevTools(m, state == "online", scope)
        Spacer(Modifier.height(8.dp))
        Dim("XTier Go ${runCatching { Native.version() }.getOrDefault("?")}", 11.sp, color = Xt.faint, mono = true)
    }
}

/** Development helpers on the device: a fixed short benchmark through the API and a prefix-cache flush. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ColumnScope.DevTools(m: LauncherModel, online: Boolean, scope: kotlinx.coroutines.CoroutineScope) {
    PanelTitle("DEV · 调试")
    Spacer(Modifier.height(8.dp))
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        for (b in LauncherModel.BENCHES)
            XtButton("测速 · ${b.label}", enabled = online && !m.benching, small = true) { scope.launch { m.bench(b) } }
        XtButton("清前缀缓存", enabled = online && !m.benching, small = true) { scope.launch { m.flushCache() } }
        XtButton("NPU 自检", enabled = m.npuTools && !m.npuChecking, small = true) { scope.launch { m.npuSelfTest() } }
    }
    Spacer(Modifier.height(4.dp))
    Dim("测速：固定提示、温度 0、关思考、先清前缀缓存；结果写进上面的日志", 10.5.sp)
    Dim("NPU 自检：用 App 进程跑 npu/ 里的上下文二进制（要带 QNN 的开发版，见 NpuProbe.kt）", 10.5.sp)
}
