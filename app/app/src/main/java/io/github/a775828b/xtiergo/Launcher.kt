package io.github.a775828b.xtiergo

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import org.json.JSONObject

// the XTier console's palette: dark blue, cyan accents
object Palette {
    val bg = Color(0xFF0B1020)
    val panel = Color(0xFF111A30)
    val line = Color(0xFF223052)
    val text = Color(0xFFDCE6FF)
    val dim = Color(0xFF7F8BA8)
    val accent = Color(0xFF3DD6F5)
    val ok = Color(0xFF5BE49B)
    val warn = Color(0xFFF5B83D)
    val bad = Color(0xFFFF6B6B)
}

@Composable
fun XTierTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Palette.accent, onPrimary = Palette.bg, background = Palette.bg, surface = Palette.panel,
            onSurface = Palette.text, onBackground = Palette.text, secondaryContainer = Color(0xFF16405A),
        ),
        content = content,
    )
}

@Composable
fun Launcher(m: LauncherModel) {
    LaunchedEffect(Unit) { m.poll() }
    Surface(Modifier.fillMaxSize(), color = Palette.bg) {
        if (m.cards.isEmpty()) {
            Text("没有模型卡片（assets/*/card.json）", color = Palette.bad, modifier = Modifier.padding(24.dp))
            return@Surface
        }
        BoxWithConstraints(Modifier.fillMaxSize().padding(12.dp)) {
            if (maxWidth >= 840.dp) {
                // tablet: models | options | status, as the XTier console
                Row(Modifier.fillMaxSize(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Panel(Modifier.width(280.dp).fillMaxHeight()) { ModelList(m) }
                    Panel(Modifier.weight(1f).fillMaxHeight()) { OptionsPane(m) }
                    Panel(Modifier.width(380.dp).fillMaxHeight()) { StatusPane(m) }
                }
            } else {
                var tab by remember { mutableIntStateOf(1) }
                Column(Modifier.fillMaxSize()) {
                    TabRow(selectedTabIndex = tab, containerColor = Palette.panel) {
                        listOf("模型", "启动参数", "运行状态").forEachIndexed { i, t ->
                            Tab(selected = tab == i, onClick = { tab = i }, text = { Text(t) })
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                    Panel(Modifier.fillMaxSize()) {
                        when (tab) {
                            0 -> ModelList(m)
                            1 -> OptionsPane(m)
                            else -> StatusPane(m)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun Panel(modifier: Modifier, content: @Composable () -> Unit) {
    Surface(modifier, color = Palette.panel, shape = RoundedCornerShape(14.dp), border = BorderStroke(1.dp, Palette.line)) {
        Box(Modifier.padding(14.dp)) { content() }
    }
}

@Composable
private fun Title(text: String) = Text(text, color = Palette.accent, fontSize = 13.sp, fontWeight = FontWeight.Bold,
    letterSpacing = 2.sp)

@Composable
private fun ModelList(m: LauncherModel) {
    Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Title("XTIER GO · 模型")
        for (c in m.cards) {
            val sel = c.id == m.selected
            val ready = sel && m.files.isNotEmpty() && m.files.all { it.state == ModelStore.FileState.Ready }
            Surface(
                Modifier.fillMaxWidth().clickable { m.select(c.id) },
                color = if (sel) Color(0xFF16284A) else Palette.bg, shape = RoundedCornerShape(10.dp),
                border = BorderStroke(1.dp, if (sel) Palette.accent else Palette.line),
            ) {
                Column(Modifier.padding(10.dp)) {
                    Text(c.name, color = Palette.text, fontWeight = FontWeight.SemiBold)
                    Text("${c.code} · :${c.port}", color = Palette.dim, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                    Text("DEC ${c.benchDec} · PRE ${c.benchPre} · ${if (c.benchKind == "agent") "AGENT" else "文本"}",
                        color = Palette.dim, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
                    if (sel) Text(if (ready) "文件齐全" else "缺文件", color = if (ready) Palette.ok else Palette.warn, fontSize = 11.sp)
                }
            }
        }
        val d = m.device
        Text("${d.soc} · 内存 ${gib(d.availBytes)} / ${gib(d.totalBytes)} 可用 · ${d.cpuFeatures}", color = Palette.dim,
            fontSize = 11.sp)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun OptionsPane(m: LauncherModel) {
    val c = m.card()
    val state = m.status?.optString("state")
    val busy = state == "loading" || state == "online"
    Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Title("启动参数")
        Text(c.name, color = Palette.text, fontSize = 18.sp, fontWeight = FontWeight.Bold)
        Text(c.arch, color = Palette.dim, fontSize = 12.sp)
        Text(c.hardware, color = Palette.dim, fontSize = 12.sp)
        FilesBox(m, c)
        for (o in c.options) {
            val v = m.values[o.key] ?: o.default
            val off = c.isOff(o, m.values)
            val reasons = o.unavailable + (m.tooBig[o.key] ?: emptyMap())
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(o.label, color = if (off) Palette.dim else Palette.text, fontWeight = FontWeight.SemiBold)
                    if (v != o.default) Text("  已改动", color = Palette.warn, fontSize = 11.sp)
                    if (off) Text("  （这个组合下不起作用）", color = Palette.dim, fontSize = 11.sp)
                }
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    for (ch in o.choices) {
                        val why = reasons[ch.value]
                        FilterChip(
                            selected = ch.value == v,
                            onClick = { if (why == null && !off) m.set(o.key, ch.value) else m.message = why ?: "" },
                            label = { Text(ch.label, fontSize = 12.sp) },
                            enabled = !off,
                            colors = FilterChipDefaults.filterChipColors(
                                labelColor = if (why != null) Palette.dim else Palette.text,
                                selectedContainerColor = Color(0xFF16405A), selectedLabelColor = Palette.accent,
                            ),
                        )
                    }
                }
                val why = reasons[v]
                if (why != null) Text("当前选择：$why", color = Palette.bad, fontSize = 11.sp)
                if (o.hint.isNotEmpty()) Text(o.hint, color = Palette.dim, fontSize = 11.sp)
            }
        }
        for (n in c.activeNotes(m.values))
            Text((if (n.warn) "⚠ " else "· ") + n.text, color = if (n.warn) Palette.warn else Palette.dim, fontSize = 12.sp)
        m.estimate?.let { e ->
            Text("预计常驻 ${gib(e.optLong("resident_bytes"))}（权重 ${gib(e.optLong("weights_bytes") - e.optLong("embedding_bytes"))}" +
                " · KV 满上下文 ${gib(e.optLong("kv_full_bytes") + e.optLong("kv_window_bytes"))}" +
                " · 快照 ${gib(e.optLong("snapshot_bytes"))}），可用 ${gib(m.device.availBytes)}",
                color = Palette.dim, fontSize = 12.sp)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Button(onClick = { m.start() }, enabled = !busy || state == "online",
                colors = ButtonDefaults.buttonColors(containerColor = Palette.accent)) {
                Text(if (state == "online") "按当前参数重启" else "启动", color = Palette.bg)
            }
            OutlinedButton(onClick = { m.stop() }, enabled = busy) { Text("停止") }
            OutlinedButton(onClick = { m.reset() }) { Text("恢复默认") }
        }
        if (m.message.isNotEmpty()) Text(m.message, color = Palette.warn, fontSize = 12.sp)
        Text("一次只跑一个模型；服务在后台进程里，关掉启动台不影响它。", color = Palette.dim, fontSize = 11.sp)
    }
}

@Composable
private fun FilesBox(m: LauncherModel, c: Card) {
    val scope = rememberCoroutineScope()
    val ready = m.files.isNotEmpty() && m.files.all { it.state == ModelStore.FileState.Ready }
    Surface(color = Palette.bg, shape = RoundedCornerShape(10.dp), border = BorderStroke(1.dp, Palette.line)) {
        Column(Modifier.fillMaxWidth().padding(10.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("模型文件 · ${c.repo}", color = Palette.text, fontSize = 13.sp)
            Text(m.store.dir(c).path, color = Palette.dim, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
            for (f in m.files) {
                val (label, color) = when (f.state) {
                    ModelStore.FileState.Ready -> "就绪" to Palette.ok
                    ModelStore.FileState.Downloading -> "下载中 ${gib(f.bytes)} / ${gib(f.size)}" to Palette.accent
                    ModelStore.FileState.Partial -> "不完整 ${gib(f.bytes)} / ${gib(f.size)}" to Palette.warn
                    ModelStore.FileState.Bad -> "大小或校验不对" to Palette.bad
                    ModelStore.FileState.Missing -> "缺" to Palette.warn
                }
                Row {
                    Text(f.name, color = Palette.dim, fontSize = 12.sp, modifier = Modifier.weight(1f))
                    Text(label, color = color, fontSize = 12.sp)
                }
            }
            if (!ready) Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (mirror in c.mirrors)
                    OutlinedButton(onClick = { m.download(mirror) }) { Text("从 ${mirror.removePrefix("https://")} 下载", fontSize = 12.sp) }
            } else OutlinedButton(onClick = { scope.launch { m.verify() } }) { Text("校验 sha256", fontSize = 12.sp) }
        }
    }
}

private val PHASES = listOf("files" to "检查文件", "tokenizer" to "分词器与模板", "weights" to "映射权重", "kv" to "分配 KV",
    "warmup" to "预热", "online" to "在线")

@Composable
private fun StatusPane(m: LauncherModel) {
    val s: JSONObject? = m.status
    val c = m.card()
    Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Title("运行状态")
        val state = s?.optString("state") ?: "stopped"
        val (label, color) = when (state) {
            "online" -> "在线" to Palette.ok
            "loading" -> (s?.optString("phase_label") ?: "加载中") to Palette.accent
            "failed" -> "失败" to Palette.bad
            else -> "未运行" to Palette.dim
        }
        Text(label, color = color, fontSize = 22.sp, fontWeight = FontWeight.Bold)
        if (state == "failed") Text(s?.optString("error").orEmpty(), color = Palette.bad, fontSize = 12.sp)
        if (state == "loading" || state == "online") {
            LinearProgressIndicator(progress = { (s?.optInt("progress") ?: 0) / 100f }, modifier = Modifier.fillMaxWidth())
            val phase = s?.optString("phase")
            val reached = PHASES.indexOfFirst { it.first == phase }
            Text(PHASES.mapIndexed { i, p -> (if (i <= reached) "● " else "○ ") + p.second }.joinToString("  "),
                color = Palette.dim, fontSize = 11.sp)
        }
        if (state == "online") {
            val t = s?.optJSONObject("telemetry") ?: JSONObject()
            val model = s?.optJSONObject("model") ?: JSONObject()
            Metric("decode", "%.1f tok/s".format(t.optDouble("decode_tok_s", 0.0)),
                "上一请求 %.1f".format(t.optDouble("last_decode_tok_s", 0.0)))
            Metric("prefill", "%.0f tok/s".format(t.optDouble("last_prefill_tok_s", 0.0)), "上一请求")
            val last = t.optJSONObject("last")
            if (last != null) Metric("上一请求", "${last.optInt("prompt_tokens")} 提示 · ${last.optInt("cached_tokens")} 命中",
                "${last.optInt("completion_tokens")} 生成 · ${last.optString("finish")}")
            Metric("请求", "${t.optInt("requests")}", if (t.optBoolean("running")) "运行中 · 排队 ${t.optInt("queued")}" else "空闲")
            Metric("内存", gib(t.optLong("rss_bytes")), "进程常驻")
            Metric("配置", "${model.optInt("loops")} 次循环 · ${model.optString("kv_mode")} · ${model.optString("kv_dtype")}",
                "上下文 ${model.optLong("context") / 1024}K · ${model.optString("kernels")} · ${model.optInt("threads")} 线程")
            Text("接口 http://127.0.0.1:${c.port}/v1 · 模型名 ${c.served}", color = Palette.accent, fontSize = 12.sp,
                fontFamily = FontFamily.Monospace)
        }
        val d = m.device
        Spacer(Modifier.height(6.dp))
        Metric("温度", "电池 %.1f °C".format(d.batteryTempC),
            "热状态 ${DeviceInfo.thermalLabel(d.thermalStatus)}" + if (!d.thermalHeadroom.isNaN()) " · 余量 %.2f".format(d.thermalHeadroom) else "")
        Metric("电量", "${d.batteryPercent}%" + if (d.charging) " · 充电中" else "",
            if (!d.batteryWatts.isNaN()) "%.1f W".format(d.batteryWatts) else "功率未知")
        Metric("内存", "可用 ${gib(d.availBytes)}", "共 ${gib(d.totalBytes)}" + if (d.lowMemory) " · 系统内存紧张" else "")
        Text("XTier Go ${runCatching { Native.version() }.getOrDefault("?")}", color = Palette.dim, fontSize = 11.sp)
    }
}

@Composable
private fun Metric(name: String, value: String, sub: String) {
    Row(Modifier.fillMaxWidth().background(Palette.bg, RoundedCornerShape(8.dp)).padding(8.dp),
        verticalAlignment = Alignment.CenterVertically) {
        Text(name, color = Palette.dim, fontSize = 12.sp, modifier = Modifier.width(72.dp))
        Column(Modifier.weight(1f)) {
            Text(value, color = Palette.text, fontSize = 14.sp, fontFamily = FontFamily.Monospace)
            Text(sub, color = Palette.dim, fontSize = 11.sp)
        }
    }
}
