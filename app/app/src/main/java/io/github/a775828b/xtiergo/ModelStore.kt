package io.github.a775828b.xtiergo

import android.content.Context
import java.io.File
import java.security.MessageDigest

/**
 * Model files live in the app's external files directory, Android/data/<package>/files/models/<card.dir>/ (no
 * storage permission needed; `adb push` can put them there too). The launcher's download button runs Downloader in
 * DownloadService: direct connections to the card's mirrors with names resolved by public DNS (not the system's,
 * which may hand out a proxy's fake IPs), resumable, checked against the sizes and sha256 in the card.
 */
class ModelStore(private val context: Context) {
    fun root(): File = File(context.getExternalFilesDir(null), "models").apply { mkdirs() }
    fun dir(card: Card): File = File(root(), card.dir)

    enum class FileState { Missing, Downloading, Partial, Ready, Bad }

    data class Status(val name: String, val state: FileState, val bytes: Long, val size: Long)

    private val verified = context.getSharedPreferences("verified", Context.MODE_PRIVATE)

    private fun stamp(file: File) = file.path + ":" + file.length() + ":" + file.lastModified()

    fun markVerified(file: File, sha: String) = verified.edit().putString(stamp(file), sha).apply()

    fun status(card: Card): List<Status> {
        val downloading = Downloads.running && Downloads.cardId == card.id
        val current = Downloads.progress?.file
        return card.files.map { f ->
            val file = File(dir(card), f.name)
            val part = File(file.path + ".part")
            val checked = verified.getString(stamp(file), null)
            when {
                file.exists() && file.length() == f.size && f.sha256 != null && checked != null && checked != f.sha256 ->
                    Status(f.name, FileState.Bad, file.length(), f.size)
                file.exists() && file.length() == f.size -> Status(f.name, FileState.Ready, f.size, f.size)
                downloading && current == f.name -> Status(f.name, FileState.Downloading, part.length(), f.size)
                file.exists() -> Status(f.name, FileState.Bad, file.length(), f.size)
                part.exists() -> Status(f.name, FileState.Partial, part.length(), f.size)
                else -> Status(f.name, FileState.Missing, 0, f.size)
            }
        }
    }

    fun ready(card: Card): Boolean = status(card).all { it.state == FileState.Ready }

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
            markVerified(file, got)
            if (got != sha) bad += f.name
        }
        return bad
    }
}
