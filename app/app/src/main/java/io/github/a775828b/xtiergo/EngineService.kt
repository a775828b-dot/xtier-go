package io.github.a775828b.xtiergo

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import androidx.core.app.ServiceCompat
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * The model server as a foreground service in its own process (:engine). The launcher starts it with the launch
 * JSON (Card.launchConfig) and follows it over http://127.0.0.1:<port>/xtg/status; closing the launcher does not
 * stop it. One model at a time: a new start stops the running one first.
 */
class EngineService : Service() {
    companion object {
        const val ACTION_START = "io.github.a775828b.xtiergo.START"
        const val ACTION_STOP = "io.github.a775828b.xtiergo.STOP"
        const val EXTRA_CONFIG = "config"
        private const val CHANNEL = "engine"
        private const val NOTIFY_ID = 1

        fun start(context: Context, config: JSONObject) {
            context.startForegroundService(
                Intent(context, EngineService::class.java).setAction(ACTION_START).putExtra(EXTRA_CONFIG, config.toString())
            )
        }

        fun stop(context: Context) {
            context.startService(Intent(context, EngineService::class.java).setAction(ACTION_STOP))
        }
    }

    private val handler = Handler(Looper.getMainLooper())
    private val commands = Executors.newSingleThreadExecutor()
    private val generation = AtomicInteger()
    private var wakeLock: PowerManager.WakeLock? = null
    @Volatile private var running = false
    @Volatile private var lastError: String? = null

    private val tick = object : Runnable {
        override fun run() {
            notifyStatus()
            if (running) handler.postDelayed(this, 3000)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                val request = generation.incrementAndGet()
                commands.execute {
                    stopEngine()
                    handler.post {
                        if (generation.get() == request) {
                            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
                            stopSelfResult(startId)
                        }
                    }
                }
            }
            ACTION_START -> {
                ensureChannel()
                ServiceCompat.startForeground(this, NOTIFY_ID, notification("加载中…"),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
                val config = intent.getStringExtra(EXTRA_CONFIG) ?: return START_NOT_STICKY
                val request = generation.incrementAndGet()
                commands.execute {
                    stopEngine()
                    if (generation.get() != request) return@execute
                    lastError = Native.start(config)
                    if (generation.get() != request) {
                        stopEngine()
                        return@execute
                    }
                    if (lastError == null) {
                        running = true
                        acquireWakeLock()
                        handler.post { if (generation.get() == request) tick.run() }
                    } else {
                        notifyStatus()
                    }
                }
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        generation.incrementAndGet()
        running = false
        handler.removeCallbacks(tick)
        commands.execute { stopEngine() }
        commands.shutdown()
        super.onDestroy()
    }

    private fun stopEngine() {
        running = false
        handler.removeCallbacks(tick)
        Native.stop()
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    // keeps the CPU up while serving (a long prefill must not stall when the screen goes off)
    private fun acquireWakeLock() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "xtiergo:engine").apply {
            setReferenceCounted(false)
            acquire()
        }
    }

    private fun ensureChannel() {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL) == null)
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "推理服务", NotificationManager.IMPORTANCE_LOW))
    }

    private fun notification(text: String): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val stop = PendingIntent.getService(this, 1, Intent(this, EngineService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle("XTier Go")
            .setContentText(text)
            .setContentIntent(open)
            .setOngoing(true)
            .addAction(Notification.Action.Builder(null, "停止", stop).build())
            .build()
    }

    private fun notifyStatus() {
        val text = lastError?.let { "启动失败：$it" } ?: try {
            val s = JSONObject(Native.status())
            val t = s.optJSONObject("telemetry")
            when (s.optString("state")) {
                "online" -> "在线 · ${s.optString("served")} · 127.0.0.1:${s.optInt("port")} · " +
                    "decode %.1f tok/s".format(t?.optDouble("last_decode_tok_s", 0.0) ?: 0.0)
                "failed" -> "失败：${s.optString("error")}"
                else -> "${s.optString("phase_label")} ${s.optInt("progress")}%"
            }
        } catch (e: Exception) {
            "状态未知"
        }
        getSystemService(NotificationManager::class.java).notify(NOTIFY_ID, notification(text))
    }
}
