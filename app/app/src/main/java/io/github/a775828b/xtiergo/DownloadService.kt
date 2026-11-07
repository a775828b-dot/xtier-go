package io.github.a775828b.xtiergo

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.wifi.WifiManager
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.ServiceCompat

/** What the launcher shows about a download (same process as the launcher: plain fields, read every second). */
object Downloads {
    @Volatile var running = false
    @Volatile var cardId = ""
    @Volatile var mirror = ""
    @Volatile var progress: Downloader.Progress? = null
    @Volatile var result = ""
    @Volatile var downloader: Downloader? = null
}

/**
 * Model download as a foreground service (dataSync), so a 3 GiB download keeps going with the screen off and the
 * launcher closed; holds a partial wake lock and a Wi-Fi lock while it runs.
 */
class DownloadService : Service() {
    companion object {
        private const val ACTION_START = "io.github.a775828b.xtiergo.DOWNLOAD"
        private const val ACTION_CANCEL = "io.github.a775828b.xtiergo.DOWNLOAD_CANCEL"
        private const val CHANNEL = "download"
        private const val NOTIFY_ID = 2

        fun start(context: Context, card: Card, mirror: String) {
            context.startForegroundService(
                Intent(context, DownloadService::class.java).setAction(ACTION_START)
                    .putExtra("card", card.id).putExtra("mirror", mirror)
            )
        }

        fun cancel(context: Context) {
            Downloads.downloader?.cancelled?.set(true)
            context.startService(Intent(context, DownloadService::class.java).setAction(ACTION_CANCEL))
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_CANCEL -> Downloads.downloader?.cancelled?.set(true)
            ACTION_START -> {
                ensureChannel()
                ServiceCompat.startForeground(this, NOTIFY_ID, notification("准备下载…", 0, 0),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
                if (Downloads.running) return START_NOT_STICKY
                val cardId = intent.getStringExtra("card").orEmpty()
                val mirror = intent.getStringExtra("mirror").orEmpty()
                val card = Card.loadAll(this).firstOrNull { it.id == cardId } ?: return START_NOT_STICKY
                Downloads.running = true
                Downloads.cardId = cardId
                Downloads.mirror = mirror
                Downloads.result = ""
                val d = Downloader(ModelStore(this))
                Downloads.downloader = d
                Thread({ work(card, mirror, d) }, "xtg-download").start()
            }
        }
        return START_NOT_STICKY
    }

    private fun work(card: Card, mirror: String, d: Downloader) {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        val wake = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "xtiergo:download").apply { acquire(6 * 3600_000L) }
        val wm = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        @Suppress("DEPRECATION")
        val wifi = wm.createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "xtiergo:download").apply { acquire() }
        var last = 0L
        try {
            val err = d.run(card, mirror) { p ->
                Downloads.progress = p
                val now = System.currentTimeMillis()
                if (now - last > 1000) {
                    last = now
                    val text = "${gib(p.done)} / ${gib(p.total)} · %.1f MB/s".format(p.bytesPerSec / 1e6)
                    getSystemService(NotificationManager::class.java)
                        .notify(NOTIFY_ID, notification(text, p.done, p.total))
                }
            }
            Downloads.result = err ?: "下载完成，已校验"
        } catch (e: Exception) {
            Downloads.result = "下载出错：${e.message}"
        } finally {
            if (wake.isHeld) wake.release()
            if (wifi.isHeld) wifi.release()
            Downloads.running = false
            Downloads.downloader = null
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun ensureChannel() {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL) == null)
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "模型下载", NotificationManager.IMPORTANCE_LOW))
    }

    private fun notification(text: String, done: Long, total: Long): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val cancel = PendingIntent.getService(this, 3, Intent(this, DownloadService::class.java).setAction(ACTION_CANCEL),
            PendingIntent.FLAG_IMMUTABLE)
        val b = Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("XTier Go · 下载模型")
            .setContentText(text)
            .setContentIntent(open)
            .setOngoing(true)
            .addAction(Notification.Action.Builder(null, "停止", cancel).build())
        if (total > 0) b.setProgress(1000, (done * 1000 / total).toInt(), false)
        return b.build()
    }
}
