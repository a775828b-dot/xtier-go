package io.github.a775828b.xtiergo

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// The rate board: what the console shows in split screen next to the app that calls the server (DSH Go X). The
// request in flight with its live rates, the decode rate over the last polls, and the finished requests (the
// server keeps the last 32 in /xtg/status), newest first.

@Composable
fun RateBoard(m: LauncherModel, onFull: () -> Unit) {
    val s = m.status
    val state = uiState(s)
    val c = m.card()
    val t = s?.optJSONObject("telemetry") ?: JSONObject()
    Console(Modifier.fillMaxSize().background(Xt.bottom).safeDrawingPadding().padding(8.dp)) {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                XtLogo()
                Column(Modifier.weight(1f)) {
                    GradientTitle("XTIER GO", 17.sp)
                    Text("RATES · 速率", color = Xt.d, fontSize = 10.5.sp, letterSpacing = .28.em, maxLines = 1)
                }
                XtButton("完整启动台", small = true, onClick = onFull)
            }
            StatePill(state, when (state) {
                "online" -> "${c.name} · :${c.port}"
                "loading" -> "${s?.optString("phase_label").orEmpty()} · ${s?.optInt("progress") ?: 0}%"
                "failed" -> s?.optString("error").orEmpty()
                else -> "待命"
            })
            // half a tablet: one column; wider (a floating window, a wide split): the two panels side by side
            BoxWithConstraints(Modifier.fillMaxWidth()) {
                if (maxWidth >= 900.dp) Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    XtPanel("LIVE · 当前请求", Modifier.weight(1f)) { Live(t, state == "online", m.spark) }
                    XtPanel("RECENT · 最近请求", Modifier.weight(1.3f)) { Recent(t) }
                } else Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    XtPanel("LIVE · 当前请求", Modifier.fillMaxWidth()) { Live(t, state == "online", m.spark) }
                    XtPanel("RECENT · 最近请求", Modifier.fillMaxWidth()) { Recent(t) }
                }
            }
            val d = m.device
            XtBar("内存 · 进程 ${if (state == "online" || state == "loading") gib(t.optLong("rss_bytes", 0L)) else "--"} · " +
                "可用 ${gib(d.availBytes)} / ${gib(d.totalBytes)}",
                if (d.totalBytes > 0) 1f - d.availBytes.toFloat() / d.totalBytes else 0f, warn = d.lowMemory)
            val head = if (d.thermalHeadroom.isNaN()) 0f else d.thermalHeadroom
            XtBar("热 · ${DeviceInfo.thermalLabel(d.thermalStatus)}" +
                (if (!d.thermalHeadroom.isNaN()) " · 余量 %.2f".format(d.thermalHeadroom) else "") +
                " · 电池 %.1f °C · %d%%".format(d.batteryTempC, d.batteryPercent), head, warn = head >= .85f)
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CodeChip("http://127.0.0.1:${c.port}/v1")
                CodeChip(c.served)
            }
        }
    }
}

@Composable
private fun ColumnScope.Live(t: JSONObject, online: Boolean, spark: List<Float>) {
    val cur = t.optJSONObject("current")
    val last = t.optJSONObject("last")
    val phase = cur?.optString("phase").orEmpty()
    // in flight: this request's rates so far; idle: the last finished request's
    val dec = when {
        phase == "decode" -> cur!!.optDouble("decode_tok_s", 0.0)
        last != null -> last.optDouble("decode_tok_s", 0.0)
        else -> 0.0
    }
    val pre = when {
        cur != null -> cur.optDouble("prefill_tok_s", 0.0)
        last != null -> last.optDouble("prefill_tok_s", 0.0)
        else -> 0.0
    }
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        BigRate("DECODE", if (online && (cur != null || last != null)) rate(dec) else "--", phase == "decode", Modifier.weight(1f))
        BigRate("PREFILL", if (online && (cur != null || last != null)) rate(pre) else "--", phase == "prefill", Modifier.weight(1f))
    }
    Spacer(Modifier.height(8.dp))
    when {
        !online -> Dim("服务没有在运行", 12.sp)
        phase == "prefill" -> {
            val done = cur!!.optLong("prefill_done")
            val todo = cur.optLong("prompt_tokens") - cur.optLong("cached_tokens")
            XtBar("prefill ${tokens(done)} / ${tokens(todo)} · 命中 ${tokens(cur.optLong("cached_tokens"))} · %.1f s"
                .format(cur.optDouble("prefill_s", 0.0)), if (todo > 0) done.toFloat() / todo else 0f)
        }
        phase == "decode" -> Dim("decode · 已生成 ${tokens(cur!!.optLong("generated"))} · %.1f s · 提示 ${tokens(cur.optLong("prompt_tokens"))}"
            .format(cur.optDouble("decode_s", 0.0)), 12.sp, mono = true)
        last != null -> Dim("空闲 · 上面是上一请求的速率", 12.sp)
        else -> Dim("空闲 · 还没有请求", 12.sp)
    }
    Spacer(Modifier.height(8.dp))
    Spark(spark)
    t.optJSONObject("spec")?.let { sp ->
        if (sp.optInt("k") > 0 && sp.optLong("steps") > 0)
            Dim("投机解码 · 每次前向 %.2f 个（草稿 ${sp.optInt("k")}）".format(sp.optDouble("tokens_per_step", 0.0)), 11.5.sp)
    }
}

@Composable
private fun BigRate(name: String, value: String, active: Boolean, modifier: Modifier) {
    val shape = RoundedCornerShape(10.dp)
    val edge = if (active) Xt.g else Xt.line
    Column(modifier.clip(shape).background(Xt.c.copy(alpha = .04f)).border(1.dp, edge, shape).padding(horizontal = 12.dp, vertical = 9.dp)) {
        Text(name, color = if (active) Xt.g else Xt.d, fontSize = 11.sp, letterSpacing = .16.em, maxLines = 1)
        Row(verticalAlignment = Alignment.Bottom) {
            Text(value, color = Color(0xFFF2FBFF), fontSize = 30.sp, fontFamily = Xt.mono, maxLines = 1)
            Text(" tok/s", color = Xt.d, fontSize = 12.sp, maxLines = 1, modifier = Modifier.padding(bottom = 4.dp))
        }
    }
}

@Composable
private fun ColumnScope.Recent(t: JSONObject) {
    val h = t.optJSONArray("history")
    val rows = (0 until (h?.length() ?: 0)).map { h!!.getJSONObject(it) }.reversed()
    TableRow(listOf("时间", "提示 / 命中", "prefill", "生成", "decode", "结束"), header = true)
    if (rows.isEmpty()) {
        Spacer(Modifier.height(6.dp))
        Dim("还没有完成的请求", 12.sp)
    }
    val clock = SimpleDateFormat("HH:mm:ss", Locale.US)
    for (r in rows.take(16)) {
        val end = r.optDouble("end", 0.0)
        TableRow(listOf(
            if (end > 0) clock.format(Date((end * 1000).toLong())) else "#${r.optInt("n")}",
            "${tokens(r.optLong("prompt_tokens"))} / ${tokens(r.optLong("cached_tokens"))}",
            if (r.optLong("prompt_tokens") > r.optLong("cached_tokens")) rate(r.optDouble("prefill_tok_s", 0.0)) else "-",
            tokens(r.optLong("completion_tokens")),
            if (r.optLong("completion_tokens") > 1) rate(r.optDouble("decode_tok_s", 0.0)) else "-",
            finishLabel(r.optString("finish")),
        ))
    }
    Spacer(Modifier.height(8.dp))
    Dim("合计 ${t.optInt("requests")} 请求 · 提示 ${tokens(t.optLong("prompt_tokens_total"))}（命中 ${tokens(t.optLong("cached_tokens_total"))}）" +
        " · 生成 ${tokens(t.optLong("generation_tokens_total"))}", 11.5.sp)
}

private val WEIGHTS = listOf(1.25f, 1.6f, 1.05f, .85f, 1.0f, .9f)

@Composable
private fun TableRow(cells: List<String>, header: Boolean = false) {
    Row(Modifier.fillMaxWidth().padding(vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
        cells.forEachIndexed { i, text ->
            Cell(text, WEIGHTS[i], if (header) Xt.d else if (i == 2 || i == 4) Xt.bright else Xt.t,
                if (header) 10.5.sp else 12.sp, header, i > 0)
        }
    }
}

@Composable
private fun RowScope.Cell(text: String, weight: Float, color: Color, size: TextUnit, header: Boolean, end: Boolean) {
    Text(text, color = color, fontSize = size, fontFamily = if (header) null else Xt.mono, maxLines = 1,
        overflow = TextOverflow.Ellipsis, letterSpacing = if (header) .06.em else 0.em,
        fontWeight = if (header) FontWeight.Normal else FontWeight.Medium,
        textAlign = if (end) TextAlign.End else TextAlign.Start, modifier = Modifier.weight(weight).padding(horizontal = 2.dp))
}

private fun rate(v: Double): String = if (v <= 0) "-" else if (v >= 100) "%.0f".format(v) else "%.1f".format(v)

private fun tokens(n: Long): String = if (n >= 10_000) "%.1fK".format(n / 1000.0) else n.toString()

private fun finishLabel(f: String): String = when (f) {
    "stop" -> "完成"
    "length" -> "长度"
    "tool_calls" -> "工具"
    "cancelled" -> "取消"
    "error" -> "出错"
    else -> f
}
