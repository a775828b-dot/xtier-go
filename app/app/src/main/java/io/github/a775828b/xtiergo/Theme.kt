package io.github.a775828b.xtiergo

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CutCornerShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp

/** The XTier console's tokens (xtier/console/ui.py): always dark, cyan / violet / pink accents. */
object Xt {
    val c = Color(0xFF3EE6FF)          // cyan
    val v = Color(0xFFA78BFA)          // violet
    val p = Color(0xFFF472B6)          // pink
    val g = Color(0xFF9CF06B)          // online
    val a = Color(0xFFFBBF24)          // loading, warnings
    val r = Color(0xFFFB7185)          // failed, errors
    val t = Color(0xFFDCECFF)          // text
    val d = Color(0xFF7F90AD)          // dim text
    val bright = Color(0xFFF4FBFF)
    val faint = Color(0xFF53607A)
    val logText = Color(0xFF8EA2C2)
    val line = Color(0x383EE6FF)       // rgba(62,230,255,.22)
    val top = Color(0xFF070B19)
    val bottom = Color(0xFF03050C)
    val panel = Color(0xB8050B1A)      // rgba(5,11,26,.72)
    val field = Color(0xE6030814)      // rgba(3,8,20,.9)
    val menu = Color(0xFF0A1226)
    val mono = FontFamily.Monospace
}

@Composable
fun XTierTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Xt.c, onPrimary = Xt.bottom, background = Xt.bottom, onBackground = Xt.t,
            surface = Xt.menu, onSurface = Xt.t, surfaceContainer = Xt.menu, onSurfaceVariant = Xt.d,
        ),
        content = content,
    )
}

/** online / loading / failed / offline, from the server's /xtg/status (null: no server). */
fun uiState(s: org.json.JSONObject?): String = when (s?.optString("state")) {
    "online" -> "online"
    "loading" -> "loading"
    "failed" -> "failed"
    else -> "offline"
}

fun stateColor(state: String): Color = when (state) {
    "online" -> Xt.g
    "loading" -> Xt.a
    "failed" -> Xt.r
    else -> Color(0xFF45506A)
}

/** The console frame: gradient glows, the faint grid and the scan line behind a rounded cyan border. */
@Composable
fun Console(modifier: Modifier, content: @Composable () -> Unit) {
    val shape = RoundedCornerShape(18.dp)
    val tr = rememberInfiniteTransition(label = "scan")
    val scan by tr.animateFloat(0f, 1f, infiniteRepeatable(tween(7000, easing = LinearEasing)), label = "scan")
    Box(
        modifier.clip(shape).border(1.dp, Xt.c.copy(alpha = .38f), shape).drawBehind {
            drawRect(Brush.verticalGradient(0f to Xt.top, .7f to Xt.bottom, 1f to Xt.bottom))
            val w = size.width
            val h = size.height
            drawRect(Brush.radialGradient(listOf(Xt.c.copy(alpha = .20f), Color.Transparent),
                center = Offset(w * .08f, -60.dp.toPx()), radius = 560.dp.toPx()))
            drawRect(Brush.radialGradient(listOf(Xt.v.copy(alpha = .22f), Color.Transparent),
                center = Offset(w * 1.05f, -30.dp.toPx()), radius = 470.dp.toPx()))
            drawRect(Brush.radialGradient(listOf(Xt.p.copy(alpha = .10f), Color.Transparent),
                center = Offset(w * .5f, h + 80.dp.toPx()), radius = 380.dp.toPx()))
            // the grid fades out towards the bottom (the CSS uses a radial mask from the top centre)
            val step = 26.dp.toPx()
            val fade = h * .85f
            val gridLine = Xt.c.copy(alpha = .055f)
            val vertical = Brush.verticalGradient(listOf(gridLine, Color.Transparent), startY = 0f, endY = fade)
            var x = 0f
            while (x < w) {
                drawLine(vertical, Offset(x, 0f), Offset(x, fade), 1f)
                x += step
            }
            var y = 0f
            while (y < fade) {
                drawLine(gridLine.copy(alpha = gridLine.alpha * (1f - y / fade)), Offset(0f, y), Offset(w, y), 1f)
                y += step
            }
            val band = h * .4f
            val top = -band + scan * h * 1.4f
            drawRect(
                Brush.verticalGradient(
                    0f to Color.Transparent, .7f to Xt.c.copy(alpha = .07f), .92f to Xt.c.copy(alpha = .16f),
                    1f to Color.Transparent, startY = top, endY = top + band,
                ),
                topLeft = Offset(0f, top), size = Size(w, band),
            )
        },
    ) { content() }
}

/** A panel with the console's "▌ TITLE" heading; scroll only where the parent bounds its height. */
@Composable
fun XtPanel(title: String, modifier: Modifier = Modifier, scroll: Boolean = false, content: @Composable ColumnScope.() -> Unit) {
    val shape = RoundedCornerShape(14.dp)
    Column(
        modifier.clip(shape).background(Xt.panel).border(1.dp, Xt.line, shape)
            .then(if (scroll) Modifier.verticalScroll(rememberScrollState()) else Modifier)
            .padding(horizontal = 16.dp, vertical = 14.dp),
    ) {
        PanelTitle(title)
        Spacer(Modifier.height(10.dp))
        content()
    }
}

@Composable
fun PanelTitle(text: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("▌", color = Xt.v, fontSize = 11.sp)
        Spacer(Modifier.width(6.dp))
        Text(text, color = Xt.c, fontSize = 11.sp, fontWeight = FontWeight.Bold, letterSpacing = .28.em)
    }
}

@Composable
fun Dim(text: String, size: TextUnit = 12.sp, color: Color = Xt.d, mono: Boolean = false, lines: Int = Int.MAX_VALUE) =
    Text(text, color = color, fontSize = size, fontFamily = if (mono) Xt.mono else null, maxLines = lines,
        overflow = TextOverflow.Ellipsis, lineHeight = size * 1.45f)

@Composable
fun XtLogo(modifier: Modifier = Modifier) {
    val tr = rememberInfiniteTransition(label = "logo")
    val rot by tr.animateFloat(0f, 360f, infiniteRepeatable(tween(14000, easing = LinearEasing)), label = "rot")
    Canvas(modifier.size(46.dp)) {
        val s = size.width / 48f
        val grad = Brush.linearGradient(listOf(Xt.c, Xt.v), start = Offset.Zero, end = Offset(size.width, size.height))
        val hex = Path().apply {
            moveTo(24 * s, 3 * s); lineTo(42 * s, 13.5f * s); lineTo(42 * s, 34.5f * s)
            lineTo(24 * s, 45 * s); lineTo(6 * s, 34.5f * s); lineTo(6 * s, 13.5f * s); close()
        }
        val mid = Offset(24 * s, 24 * s)
        drawCircle(Brush.radialGradient(listOf(Xt.c.copy(alpha = .35f), Color.Transparent), mid, 26 * s), 26 * s, mid)
        drawPath(hex, Xt.c.copy(alpha = .08f))
        drawPath(hex, grad, style = Stroke(2 * s))
        rotate(rot, mid) {
            drawCircle(Xt.c.copy(alpha = .8f), 13 * s, mid,
                style = Stroke(1.2f * s, pathEffect = PathEffect.dashPathEffect(floatArrayOf(3 * s, 4 * s))))
        }
        drawLine(grad, Offset(16 * s, 16 * s), Offset(32 * s, 32 * s), 3.2f * s, StrokeCap.Round)
        drawLine(grad, Offset(32 * s, 16 * s), Offset(16 * s, 32 * s), 3.2f * s, StrokeCap.Round)
        drawCircle(Color(0xFFE9FBFF), 3 * s, mid)
    }
}

@Composable
fun GradientTitle(text: String, size: TextUnit = 19.sp) = Text(
    text,
    style = TextStyle(
        brush = Brush.linearGradient(listOf(Color(0xFFE9FBFF), Xt.c, Xt.v)),
        fontSize = size, fontWeight = FontWeight.ExtraBold, letterSpacing = .22.em,
    ),
)

/** The status lamp: green pulse online, amber blink while loading, red when failed, grey otherwise. */
@Composable
fun Led(state: String, modifier: Modifier = Modifier) {
    val tr = rememberInfiniteTransition(label = "led")
    val pulse by tr.animateFloat(0f, 1f, infiniteRepeatable(tween(900, easing = LinearEasing), RepeatMode.Reverse), label = "pulse")
    val blink by tr.animateFloat(0f, 1f, infiniteRepeatable(tween(900, easing = LinearEasing)), label = "blink")
    val color = stateColor(state)
    Canvas(modifier.size(24.dp)) {
        val r = 5.dp.toPx()
        val glow = when (state) {
            "online" -> (10 + 10 * pulse).dp.toPx()
            "loading", "failed" -> 10.dp.toPx()
            else -> 0f
        }
        if (glow > 0) drawCircle(Brush.radialGradient(listOf(color.copy(alpha = .65f), Color.Transparent), center, r + glow), r + glow)
        else drawCircle(color.copy(alpha = .25f), r + 3.dp.toPx())
        drawCircle(color.copy(alpha = if (state == "loading" && blink > .5f) .35f else 1f), r)
    }
}

@Composable
fun StatePill(state: String, sub: String, modifier: Modifier = Modifier) {
    val word = when (state) {
        "online" -> "ONLINE"
        "loading" -> "LOADING"
        "failed" -> "FAILED"
        else -> "OFFLINE"
    }
    val shape = RoundedCornerShape(999.dp)
    Row(
        modifier.clip(shape).background(Color(0x99040A18)).border(1.dp, Xt.line, shape).padding(start = 8.dp, end = 14.dp, top = 4.dp, bottom = 4.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Led(state)
        Text(word, color = if (state == "offline") Xt.t else stateColor(state), fontSize = 13.sp, fontWeight = FontWeight.Bold,
            letterSpacing = .2.em, fontFamily = Xt.mono)
        Dim(sub, 12.sp, lines = 1)
    }
}

/** 启动 · ENGAGE: the gradient button with cut corners and a passing shine. */
@Composable
fun EngageButton(text: String, enabled: Boolean, onClick: () -> Unit) {
    val shape = CutCornerShape(topStart = 12.dp, bottomEnd = 12.dp)
    val tr = rememberInfiniteTransition(label = "shine")
    val x by tr.animateFloat(0f, 1f, infiniteRepeatable(tween(3200, easing = LinearEasing)), label = "x")
    Box(
        Modifier.alpha(if (enabled) 1f else .38f).clip(shape)
            .background(Brush.linearGradient(listOf(Xt.c, Color(0xFF7AD8FF), Xt.v)))
            .clickable(enabled = enabled, onClick = onClick)
            .drawWithContent {
                drawContent()
                if (enabled && x < .6f) {
                    val w = size.width
                    val left = (-.6f + x / .6f * 1.9f) * w
                    val bw = .4f * w
                    val skew = size.height * .36f
                    val band = Path().apply {
                        moveTo(left + skew, 0f); lineTo(left + bw + skew, 0f); lineTo(left + bw, size.height)
                        lineTo(left, size.height); close()
                    }
                    drawPath(band, Brush.horizontalGradient(listOf(Color.Transparent, Color.White.copy(alpha = .6f), Color.Transparent),
                        startX = left, endX = left + bw + skew))
                }
            }
            .padding(horizontal = 26.dp, vertical = 12.dp),
    ) {
        Text(text, color = Color(0xFF021018), fontSize = 15.sp, fontWeight = FontWeight.Bold, letterSpacing = .18.em)
    }
}

/** The console's outlined button; danger = the red 停止 · ABORT. */
@Composable
fun XtButton(text: String, enabled: Boolean = true, danger: Boolean = false, small: Boolean = false, onClick: () -> Unit) {
    val shape = RoundedCornerShape(10.dp)
    val color = if (danger) Xt.r else Xt.t
    Box(
        Modifier.alpha(if (enabled) 1f else .38f).clip(shape)
            .background(if (danger) Xt.r.copy(alpha = .06f) else Xt.c.copy(alpha = .05f))
            .border(1.dp, if (danger) Xt.r.copy(alpha = .45f) else Xt.line, shape)
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = if (small) 12.dp else 16.dp, vertical = if (small) 7.dp else 10.dp),
    ) {
        Text(text, color = color, fontSize = if (small) 12.sp else 13.sp, fontWeight = FontWeight.Bold, letterSpacing = .08.em)
    }
}

data class Pick(val value: String, val label: String, val blocked: String?)

/** A launch option as the console's select: changed = violet with "已改动 · 非实测默认", blocked choices greyed. */
@Composable
fun SelectField(
    label: String, picks: List<Pick>, current: String, changed: Boolean, off: Boolean, hint: String,
    modifier: Modifier = Modifier, onPick: (String) -> Unit,
) {
    var open by remember { mutableStateOf(false) }
    val shape = RoundedCornerShape(9.dp)
    val now = picks.firstOrNull { it.value == current }
    Column(modifier.alpha(if (off) .42f else 1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Dim(label, 12.sp)
        Box {
            Row(
                Modifier.fillMaxWidth().clip(shape).background(Xt.field)
                    .border(1.dp, if (changed) Xt.v else Xt.c.copy(alpha = .28f), shape)
                    .clickable(enabled = !off) { open = true }
                    .padding(start = 10.dp, end = 10.dp, top = 8.dp, bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(now?.label ?: current, color = Xt.t, fontSize = 13.sp, modifier = Modifier.weight(1f), maxLines = 1,
                    overflow = TextOverflow.Ellipsis)
                Canvas(Modifier.size(12.dp, 8.dp)) {
                    val p = Path().apply { moveTo(1.dp.toPx(), 1.dp.toPx()); lineTo(6.dp.toPx(), 6.dp.toPx()); lineTo(11.dp.toPx(), 1.dp.toPx()) }
                    drawPath(p, Xt.c, style = Stroke(1.6.dp.toPx()))
                }
            }
            DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                for (p in picks) {
                    DropdownMenuItem(
                        text = {
                            Column {
                                Text(p.label + if (p.blocked != null) " · 本机不可用" else "", fontSize = 13.sp,
                                    color = when {
                                        p.blocked != null -> Xt.faint
                                        p.value == current -> Xt.c
                                        else -> Xt.t
                                    })
                                if (p.blocked != null) Dim(p.blocked, 11.sp)
                            }
                        },
                        enabled = p.blocked == null,
                        onClick = { open = false; onPick(p.value) },
                    )
                }
            }
        }
        if (changed) Text("已改动 · 非实测默认", color = Xt.v, fontSize = 10.5.sp)
        if (off) Dim("这个组合下不起作用", 10.5.sp)
        val why = now?.blocked
        if (why != null) Text("当前选择：$why", color = Xt.r, fontSize = 10.5.sp)
        if (hint.isNotEmpty()) Dim(hint, 10.5.sp)
    }
}

/** An on / off option (the card's switch: its first choice is "on"). */
@Composable
fun SwitchField(label: String, on: Boolean, text: String, changed: Boolean, off: Boolean, hint: String,
                modifier: Modifier = Modifier, onToggle: () -> Unit) {
    val shape = RoundedCornerShape(9.dp)
    val knob by animateFloatAsState(if (on) 1f else 0f, tween(180), label = "knob")
    Column(modifier.alpha(if (off) .42f else 1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Dim(label, 12.sp)
        Row(
            Modifier.fillMaxWidth().clip(shape).background(Xt.field)
                .border(1.dp, if (changed) Xt.v else Xt.c.copy(alpha = .28f), shape)
                .clickable(enabled = !off, onClick = onToggle).padding(horizontal = 10.dp, vertical = 7.dp),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Canvas(Modifier.size(38.dp, 20.dp)) {
                val rad = androidx.compose.ui.geometry.CornerRadius(size.height / 2)
                if (on) drawRoundRect(Brush.horizontalGradient(listOf(Xt.c, Xt.v)), cornerRadius = rad)
                else drawRoundRect(Xt.d.copy(alpha = .32f), cornerRadius = rad)
                drawCircle(Xt.t, 7.dp.toPx(), Offset((10 + 18 * knob).dp.toPx(), size.height / 2))
            }
            Text(text, color = Xt.t, fontSize = 13.sp)
        }
        if (changed) Text("已改动 · 非实测默认", color = Xt.v, fontSize = 10.5.sp)
        if (hint.isNotEmpty()) Dim(hint, 10.5.sp)
    }
}

@Composable
fun Meter(name: String, value: String, unit: String, modifier: Modifier = Modifier) {
    val shape = RoundedCornerShape(9.dp)
    Column(modifier.clip(shape).background(Xt.c.copy(alpha = .035f)).border(1.dp, Xt.line, shape).padding(horizontal = 9.dp, vertical = 7.dp)) {
        Text(name, color = Xt.d, fontSize = 10.5.sp, letterSpacing = .12.em, maxLines = 1)
        Row(verticalAlignment = Alignment.Bottom) {
            Text(value, color = Color(0xFFF2FBFF), fontSize = 17.sp, fontFamily = Xt.mono, maxLines = 1)
            if (unit.isNotEmpty()) Text(" $unit", color = Xt.d, fontSize = 11.sp, maxLines = 1)
        }
    }
}

/** The loading / online ring: track, tick marks, the progress arc (gradient, green when online), a big number. */
@Composable
fun Ring(progress: Float, online: Boolean, big: String, sub: String, size: Dp = 150.dp) {
    val shown by animateFloatAsState(progress.coerceIn(0f, 1f), tween(800), label = "ring")
    Box(Modifier.size(size), contentAlignment = Alignment.Center) {
        Canvas(Modifier.fillMaxSize()) {
            val s = this.size.width / 150f
            val r = 62 * s
            val tl = Offset(center.x - r, center.y - r)
            drawCircle(Xt.c.copy(alpha = .12f), r, style = Stroke(8 * s))
            drawCircle(Xt.c.copy(alpha = .35f), 71 * s, style = Stroke(2 * s, pathEffect = PathEffect.dashPathEffect(floatArrayOf(1 * s, 7.6f * s))))
            if (shown > 0f) {
                val brush = if (online) SolidColor(Xt.g) else Brush.sweepGradient(listOf(Xt.c, Xt.v, Xt.p, Xt.c))
                val glow = (if (online) Xt.g else Xt.c).copy(alpha = .18f)
                drawArc(glow, -90f, 360f * shown, false, tl, Size(2 * r, 2 * r), style = Stroke(16 * s, cap = StrokeCap.Round))
                drawArc(brush, -90f, 360f * shown, false, tl, Size(2 * r, 2 * r), style = Stroke(8 * s, cap = StrokeCap.Round))
            }
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(big, color = Color.White, fontSize = 30.sp, fontFamily = Xt.mono, maxLines = 1)
            Text(sub, color = Xt.d, fontSize = 10.5.sp, letterSpacing = .2.em, maxLines = 1)
        }
    }
}

/** The phase chips: passed ones cyan, the current one amber (green when online). */
@OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
@Composable
fun Phases(phases: List<Pair<String, String>>, current: String?, online: Boolean) {
    val at = phases.indexOfFirst { it.first == current }
    androidx.compose.foundation.layout.FlowRow(horizontalArrangement = Arrangement.spacedBy(5.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
        phases.forEachIndexed { i, (_, label) ->
            val shape = RoundedCornerShape(6.dp)
            val nowColor = if (online) Xt.g else Xt.a
            val (fg, bg, border) = when {
                i == at -> Triple(Color(0xFF03121A), nowColor, nowColor)
                at >= 0 && i < at -> Triple(Xt.c, Color.Transparent, Xt.c.copy(alpha = .35f))
                else -> Triple(Xt.faint, Color.Transparent, Xt.d.copy(alpha = .25f))
            }
            Text(label, color = fg, fontSize = 10.5.sp, letterSpacing = .04.em, maxLines = 1,
                modifier = Modifier.clip(shape).background(bg).border(1.dp, border, shape).padding(horizontal = 6.dp, vertical = 2.dp))
        }
    }
}

/** The decode rate over the last polls. */
@Composable
fun Spark(values: List<Float>, modifier: Modifier = Modifier) {
    val shape = RoundedCornerShape(9.dp)
    Canvas(modifier.fillMaxWidth().height(46.dp).clip(shape)
        .background(Brush.verticalGradient(listOf(Xt.c.copy(alpha = .06f), Color.Transparent))).border(1.dp, Xt.line, shape)) {
        if (values.size < 2) return@Canvas
        val top = values.max().coerceAtLeast(1f)
        val dx = size.width / (values.size - 1)
        val pad = 4.dp.toPx()
        fun y(v: Float) = size.height - pad - (size.height - 2 * pad) * (v / top)
        val line = Path()
        values.forEachIndexed { i, v -> if (i == 0) line.moveTo(0f, y(v)) else line.lineTo(i * dx, y(v)) }
        val area = Path().apply { addPath(line); lineTo(size.width, size.height); lineTo(0f, size.height); close() }
        drawPath(area, Xt.c.copy(alpha = .12f))
        drawPath(line, Xt.c, style = Stroke(1.6.dp.toPx()))
    }
}

/** A labelled bar (memory, heat), the console's GPU bar. */
@Composable
fun XtBar(text: String, fraction: Float, modifier: Modifier = Modifier, warn: Boolean = false) {
    Column(modifier) {
        Dim(text, 11.5.sp, lines = 1)
        Spacer(Modifier.height(4.dp))
        Box(Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(999.dp)).background(Xt.c.copy(alpha = .1f))) {
            Box(Modifier.fillMaxWidth(fraction.coerceIn(0f, 1f)).height(6.dp)
                .background(if (warn) Brush.horizontalGradient(listOf(Xt.a, Xt.r)) else Brush.horizontalGradient(listOf(Xt.c, Xt.v, Xt.p))))
        }
    }
}

@Composable
fun CodeChip(text: String) {
    val shape = RoundedCornerShape(7.dp)
    Text(text, color = Xt.c, fontSize = 12.sp, fontFamily = Xt.mono,
        modifier = Modifier.clip(shape).background(Xt.field).border(1.dp, Xt.line, shape).padding(horizontal = 8.dp, vertical = 3.dp))
}

/** The console's log: the last lines, a blinking cursor after the newest. */
@Composable
fun LogBox(lines: List<String>, modifier: Modifier = Modifier, rows: Int = 7) {
    val shape = RoundedCornerShape(9.dp)
    val tr = rememberInfiniteTransition(label = "cursor")
    val blink by tr.animateFloat(0f, 1f, infiniteRepeatable(tween(1000, easing = LinearEasing)), label = "blink")
    Column(modifier.fillMaxWidth().clip(shape).background(Color(0xD902050C)).border(1.dp, Xt.d.copy(alpha = .18f), shape)
        .padding(horizontal = 10.dp, vertical = 8.dp)) {
        val shown = lines.takeLast(rows).ifEmpty { listOf("待命") }
        shown.forEachIndexed { i, l ->
            Text(l + if (i == shown.lastIndex && blink < .5f) " ▋" else "", color = Xt.logText, fontSize = 11.sp,
                fontFamily = Xt.mono, maxLines = 1, overflow = TextOverflow.Ellipsis, lineHeight = 16.sp)
        }
    }
}

/** The model card in the rail: cut corners, gradient edge (bright when chosen), README speeds. */
@Composable
fun ModelTab(card: Card, selected: Boolean, running: String, modifier: Modifier = Modifier, onClick: () -> Unit) {
    val shape = CutCornerShape(topStart = 9.dp, bottomEnd = 9.dp)
    val edge = if (selected) Brush.linearGradient(listOf(Xt.c, Xt.v, Xt.p))
    else Brush.linearGradient(listOf(Xt.c.copy(alpha = .32f), Xt.v.copy(alpha = .16f), Xt.c.copy(alpha = .07f)))
    val inner = if (selected) Brush.verticalGradient(listOf(Color(0xF70E1E3A), Color(0xF7060C1C)))
    else Brush.verticalGradient(listOf(Color(0xF5091124), Color(0xF5050914)))
    Box(modifier.clip(shape).background(edge).padding(1.dp).clip(shape).background(inner).clickable(onClick = onClick)
        .padding(horizontal = 11.dp, vertical = 8.dp)) {
        Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("${card.code} · :${card.port}", color = Xt.c, fontSize = 9.5.sp, letterSpacing = .18.em, fontFamily = Xt.mono,
                    maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                if (running.isNotEmpty()) {
                    val bg = if (running == "online") Xt.g else Xt.a
                    Text(if (running == "online") "RUNNING" else "LOADING", color = Color(0xFF04110A), fontSize = 8.5.sp,
                        letterSpacing = .16.em, fontFamily = Xt.mono,
                        modifier = Modifier.clip(RoundedCornerShape(999.dp)).background(bg).padding(horizontal = 6.dp))
                }
            }
            Text(card.name, color = Xt.bright, fontSize = 13.5.sp, fontWeight = FontWeight.Bold, maxLines = 1,
                overflow = TextOverflow.Ellipsis)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    PerfLine("DEC", card.benchDec)
                    PerfLine("PRE", card.benchPre)
                }
                val agent = card.benchKind == "agent"
                val kc = if (agent) Xt.g else Xt.d
                Text(if (agent) "AGENT" else "文本", color = kc, fontSize = 8.5.sp, letterSpacing = .12.em,
                    modifier = Modifier.border(1.dp, kc.copy(alpha = .5f), RoundedCornerShape(5.dp)).padding(horizontal = 5.dp))
            }
        }
    }
}

@Composable
private fun PerfLine(name: String, value: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(name, color = Xt.d, fontSize = 9.5.sp, letterSpacing = .12.em, modifier = Modifier.width(32.dp))
        Text(value, color = Color(0xFFE3F3FF), fontSize = 11.sp, fontFamily = Xt.mono, fontWeight = FontWeight.SemiBold, maxLines = 1)
    }
}
