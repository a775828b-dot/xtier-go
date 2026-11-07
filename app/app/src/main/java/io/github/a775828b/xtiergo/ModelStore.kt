package io.github.a775828b.xtiergo

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import java.io.File
import java.security.MessageDigest

/**
 * Model files live in the app's external files directory, Android/data/<package>/files/models/<card.dir>/ (no
 * storage permission needed; `adb push` can put them there too). Downloads go through the system DownloadManager
 * (it resumes on its own) from the card's mirrors; files with a sha256 are checked after the download.
 */
class ModelStore(private val context: Context) {
    fun root(): File = File(context.getExternalFilesDir(null), "models").apply { mkdirs() }
    fun dir(card: Card): File = File(root(), card.dir)

    enum class FileState { Missing, Downloading, Partial, Ready, Bad }

    data class Status(val name: String, val state: FileState, val bytes: Long, val size: Long)

    private val prefs = context.getSharedPreferences("downloads", Context.MODE_PRIVATE)
    private val verified = context.getSharedPreferences("verified", Context.MODE_PRIVATE)

    fun status(card: Card): List<Status> {
        val dm = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        return card.files.map { f ->
            val file = File(dir(card), f.name)
            val id = prefs.getLong(key(card, f), -1L)
            if (id >= 0) {
                dm.query(DownloadManager.Query().setFilterById(id)).use { c ->
                    if (c.moveToFirst()) {
                        val st = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
                        val done = c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                        if (st == DownloadManager.STATUS_RUNNING || st == DownloadManager.STATUS_PENDING ||
                            st == DownloadManager.STATUS_PAUSED
                        ) return@map Status(f.name, FileState.Downloading, done, f.size)
                    }
                }
            }
            when {
                !file.exists() -> Status(f.name, FileState.Missing, 0, f.size)
                file.length() < f.size -> Status(f.name, FileState.Partial, file.length(), f.size)
                file.length() != f.size -> Status(f.name, FileState.Bad, file.length(), f.size)
                f.sha256 != null && verified.getString(file.path + ":" + file.lastModified(), null) != f.sha256 &&
                    verified.getString(file.path + ":" + file.lastModified(), null) != null ->
                    Status(f.name, FileState.Bad, file.length(), f.size)
                else -> Status(f.name, FileState.Ready, file.length(), f.size)
            }
        }
    }

    fun ready(card: Card): Boolean = status(card).all { it.state == FileState.Ready }

    /** Queue the missing files (mirror: one of card.mirrors). */
    fun download(card: Card, mirror: String) {
        val dm = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        val dir = dir(card).apply { mkdirs() }
        for ((f, st) in card.files.zip(status(card))) {
            if (st.state == FileState.Ready || st.state == FileState.Downloading) continue
            File(dir, f.name).delete()
            val url = "$mirror/${card.repo}/resolve/${card.revision}/${f.name}"
            val req = DownloadManager.Request(Uri.parse(url))
                .setTitle("XTier Go · ${f.name}")
                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
                .setDestinationInExternalFilesDir(context, null, "models/${card.dir}/${f.name}")
                .setAllowedOverMetered(true)
            prefs.edit().putLong(key(card, f), dm.enqueue(req)).apply()
        }
    }

    /** sha256 of the files that carry one (slow for the weights: call off the main thread). */
    fun verify(card: Card): List<String> {
        val bad = mutableListOf<String>()
        for (f in card.files) {
            val sha = f.sha256 ?: continue
            val file = File(dir(card), f.name)
            if (!file.exists()) continue
            val md = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buf = ByteArray(1 shl 20)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    md.update(buf, 0, n)
                }
            }
            val got = md.digest().joinToString("") { "%02x".format(it) }
            verified.edit().putString(file.path + ":" + file.lastModified(), got).apply()
            if (got != sha) bad += f.name
        }
        return bad
    }

    private fun key(card: Card, f: ModelFile) = "${card.id}/${f.name}"
}
