package io.github.a775828b.xtiergo

import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
class EngineServiceTest {
    private fun status(port: Int): JSONObject? = runCatching {
        val conn = URL("http://127.0.0.1:$port/xtg/status").openConnection() as HttpURLConnection
        conn.connectTimeout = 500
        conn.readTimeout = 1000
        try { conn.inputStream.bufferedReader().use { JSONObject(it.readText()) } }
        finally { conn.disconnect() }
    }.getOrNull()

    private fun awaitState(description: String, timeout: Long = 30, predicate: () -> Boolean) {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(timeout)
        while (System.nanoTime() < deadline) {
            if (predicate()) return
            Thread.sleep(100)
        }
        fail("Timed out: $description")
    }

    @Test fun stoppingPrefillUnblocksActiveAndQueuedRequestsAndAllowsRestart() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val ctx = instrumentation.targetContext
        val card = Card.loadAll(ctx).first()
        val dir = ModelStore(ctx).dir(card)
        assumeTrue("Requires real Dory checkpoint", File(dir, "model.safetensors").length() ==
            card.files.first { it.name == "model.safetensors" }.size)
        instrumentation.startActivitySync(Intent(ctx, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        val cfg = card.launchConfig(dir.path, card.defaults() + mapOf("loops" to "1"))
        cfg.getJSONObject("options").put("context", "4096").put("prefill_chunk", "64")
        cfg.put("warmup", false)
        val clients = Executors.newFixedThreadPool(2)
        fun post(): Int {
            val conn = URL("http://127.0.0.1:${card.port}/v1/chat/completions").openConnection() as HttpURLConnection
            conn.connectTimeout = 2000
            conn.readTimeout = 30000
            conn.requestMethod = "POST"
            conn.doOutput = true
            conn.setRequestProperty("Content-Type", "application/json")
            val body = JSONObject().put("model", card.served).put("max_tokens", 32)
                .put("messages", JSONArray().put(JSONObject().put("role", "user")
                    .put("content", "Explain this document. " + "The quick brown fox jumps over the lazy dog. ".repeat(160))))
            return try {
                conn.outputStream.use { it.write(body.toString().toByteArray()) }
                conn.responseCode
            } finally { conn.disconnect() }
        }
        try {
            instrumentation.runOnMainSync { EngineService.start(ctx, cfg) }
            awaitState("online") { status(card.port)?.optString("state") == "online" }
            val active = clients.submit<Int> { post() }
            awaitState("active request") { status(card.port)?.optJSONObject("telemetry")?.optBoolean("running") == true }
            val queued = clients.submit<Int> { post() }
            awaitState("queued request") { (status(card.port)?.optJSONObject("telemetry")?.optInt("queued") ?: 0) > 0 }
            val start = System.nanoTime()
            instrumentation.runOnMainSync { EngineService.stop(ctx) }
            assertTrue("Stop blocked the main thread", System.nanoTime() - start < TimeUnit.SECONDS.toNanos(1))
            assertEquals(500, active.get(30, TimeUnit.SECONDS).toInt())
            assertEquals(500, queued.get(30, TimeUnit.SECONDS).toInt())
            awaitState("server stopped") { status(card.port) == null }
            instrumentation.runOnMainSync { EngineService.start(ctx, cfg) }
            awaitState("restart online") { status(card.port)?.optString("state") == "online" }
        } finally {
            instrumentation.runOnMainSync { EngineService.stop(ctx) }
            clients.shutdownNow()
            awaitState("cleanup") { status(card.port) == null }
        }
    }
}
