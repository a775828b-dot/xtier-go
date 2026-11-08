package io.github.a775828b.xtiergo

import okhttp3.Dns
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.net.InetAddress
import java.net.Proxy
import java.net.UnknownHostException
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Names resolved by public DNS over HTTPS (AliDNS 223.5.5.5, then Tencent HTTPDNS 119.29.29.29), never by the
 * system resolver: a home network that answers fake IPs of a proxy (mihomo fake-ip) would route the model through
 * the proxy. The connection then goes straight to the real address (TLS still checks the certificate for the name).
 */
object DirectDns : Dns {
    // only IP-literal URLs below, so this client needs no DNS itself
    private val boot = OkHttpClient.Builder().proxy(Proxy.NO_PROXY)
        .connectTimeout(5, TimeUnit.SECONDS).readTimeout(5, TimeUnit.SECONDS).build()
    private val cache = ConcurrentHashMap<String, Pair<Long, List<InetAddress>>>()

    @Volatile var lastAnswer = ""
        private set

    override fun lookup(hostname: String): List<InetAddress> {
        if (hostname.all { it.isDigit() || it == '.' }) return listOf(InetAddress.getByName(hostname))
        cache[hostname]?.let { if (System.currentTimeMillis() - it.first < 600_000) return it.second }
        var via = "223.5.5.5"
        var ips = aliDns(hostname)
        if (ips.isEmpty()) {
            via = "119.29.29.29"
            ips = tencentDns(hostname)
        }
        if (ips.isEmpty()) throw UnknownHostException("$hostname: 223.5.5.5 和 119.29.29.29 都没有答复")
        val addrs = ips.map { InetAddress.getByName(it) }   // literals: no lookup
        cache[hostname] = System.currentTimeMillis() to addrs
        lastAnswer = "$hostname → ${ips.first()}（$via）"
        return addrs
    }

    private fun aliDns(host: String): List<String> = runCatching {
        boot.newCall(Request.Builder().url("https://223.5.5.5/resolve?name=$host&type=1").build()).execute().use { r ->
            val answer: JSONArray = JSONObject(r.body?.string().orEmpty()).optJSONArray("Answer") ?: JSONArray()
            (0 until answer.length()).map { answer.getJSONObject(it) }.filter { it.optInt("type") == 1 }.map { it.getString("data") }
        }
    }.getOrDefault(emptyList())

    private fun tencentDns(host: String): List<String> = runCatching {
        boot.newCall(Request.Builder().url("http://119.29.29.29/d?dn=$host").build()).execute().use { r ->
            r.body?.string().orEmpty().trim().split(';').filter { s -> s.isNotBlank() && s.all { it.isDigit() || it == '.' } }
        }
    }.getOrDefault(emptyList())
}

/**
 * Downloads a card's model files into ModelStore.dir(card): small files in one request, large ones in 16 MiB ranges
 * over 8 connections (the CDN behind hf-mirror gave one connection about 1.2 MB/s, 2026-11-07), resumable (<file>.part and
 * <file>.part.json list the finished ranges), each file checked against its size and, when the card has one, its
 * sha256 before it is renamed into place.
 */
class Downloader(private val store: ModelStore) {
    data class Progress(
        val file: String,
        val done: Long,            // bytes of all files
        val total: Long,
        val bytesPerSec: Double,
        val note: String,
    )

    val cancelled = AtomicBoolean(false)
    private val client = OkHttpClient.Builder()
        .proxy(Proxy.NO_PROXY)
        .dns(DirectDns)
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .followRedirects(true)
        .followSslRedirects(true)
        .build()

    /** null when every file is in place and checked, else what went wrong. Local files (no source) are skipped. */
    fun run(card: Card, mirror: String, onProgress: (Progress) -> Unit): String? {
        val dir = store.dir(card).apply { mkdirs() }
        val files = card.files.filter { !it.local }
        val total = files.sumOf { it.size }
        val doneFiles = AtomicLong(files.filter { File(dir, it.name).length() == it.size }.sumOf { it.size })
        val started = System.nanoTime()
        val fresh = AtomicLong(0)   // bytes fetched in this run (for the speed)
        for (f in files) {
            if (cancelled.get()) return "已停止"
            val dest = File(dir, f.name)
            if (dest.exists() && dest.length() == f.size) {
                if (f.sha256 == null) continue
                onProgress(Progress(f.name, doneFiles.get(), total, 0.0, "校验已有文件…"))
                val got = sha256(dest)
                store.markVerified(dest, got)
                if (got == f.sha256) continue
                doneFiles.addAndGet(-f.size)
            }
            dest.delete()
            val url = "$mirror/${card.repo}/resolve/${card.revision}/${f.name}"
            val report = { inFile: Long ->
                val secs = (System.nanoTime() - started) / 1e9
                onProgress(Progress(f.name, doneFiles.get() + inFile, total, if (secs > 0) fresh.get() / secs else 0.0, DirectDns.lastAnswer))
            }
            val err = try {
                if (f.size >= (64L shl 20)) ranged(url, dest, f.size, fresh, report) else whole(url, dest, f.size, fresh, report)
            } catch (e: IOException) {
                "${f.name}: ${e.message}"
            }
            if (err != null) return err
            if (f.sha256 != null) {
                onProgress(Progress(f.name, doneFiles.get() + f.size, total, 0.0, "校验 sha256…"))
                val got = sha256(dest)
                if (got != f.sha256) {
                    dest.delete()
                    return "${f.name}: sha256 不对（$got），已删掉，重新下载"
                }
                store.markVerified(dest, got)
            }
            doneFiles.addAndGet(f.size)
        }
        onProgress(Progress("", total, total, 0.0, "完成"))
        return null
    }

    private fun whole(url: String, dest: File, size: Long, fresh: AtomicLong, report: (Long) -> Unit): String? {
        val part = File(dest.path + ".part")
        client.newCall(Request.Builder().url(url).build()).execute().use { r ->
            if (!r.isSuccessful) return "${dest.name}: HTTP ${r.code}"
            val body = r.body ?: return "${dest.name}: 空的回复"
            part.outputStream().use { out ->
                val buf = ByteArray(1 shl 16)
                var got = 0L
                val input = body.byteStream()
                while (true) {
                    if (cancelled.get()) return "已停止"
                    val n = input.read(buf)
                    if (n < 0) break
                    out.write(buf, 0, n)
                    got += n
                    fresh.addAndGet(n.toLong())
                    report(got)
                }
            }
        }
        if (part.length() != size) return "${dest.name}: 收到 ${part.length()} 字节，应为 $size"
        return if (part.renameTo(dest)) null else "${dest.name}: 改名失败"
    }

    private fun ranged(url: String, dest: File, size: Long, fresh: AtomicLong, report: (Long) -> Unit): String? {
        val part = File(dest.path + ".part")
        val state = File(dest.path + ".part.json")
        val chunk = 16L shl 20
        val n = ((size + chunk - 1) / chunk).toInt()
        val finished = BooleanArray(n)
        runCatching {
            val j = JSONObject(state.readText())
            if (j.getLong("size") == size && part.length() == size) {
                val a = j.getJSONArray("done")
                for (i in 0 until a.length()) finished[a.getInt(i)] = true
            }
        }
        RandomAccessFile(part, "rw").use { it.setLength(size) }
        val inFile = AtomicLong(finished.count { it } * chunk)
        val next = AtomicLong(0)
        val errors = ConcurrentHashMap<Int, String>()
        val lock = Any()
        fun save() = synchronized(lock) {
            val done = JSONArray()
            finished.forEachIndexed { i, ok -> if (ok) done.put(i) }
            state.writeText(JSONObject().put("size", size).put("done", done).toString())
        }
        val workers = (1..8).map {
            Thread {
                RandomAccessFile(part, "rw").use { raf ->
                    val buf = ByteArray(1 shl 16)
                    while (!cancelled.get()) {
                        val i = next.getAndIncrement().toInt()
                        if (i >= n) break
                        if (finished[i]) continue
                        val a = i * chunk
                        val b = minOf(size, a + chunk) - 1
                        var ok = false
                        for (attempt in 1..5) {
                            if (cancelled.get()) break
                            try {
                                val req = Request.Builder().url(url).header("Range", "bytes=$a-$b").build()
                                client.newCall(req).execute().use { r ->
                                    if (r.code != 206) throw IOException("HTTP ${r.code}（不支持分段？）")
                                    val input = r.body!!.byteStream()
                                    var pos = a
                                    while (true) {
                                        val k = input.read(buf)
                                        if (k < 0) break
                                        raf.seek(pos)
                                        raf.write(buf, 0, k)
                                        pos += k
                                        fresh.addAndGet(k.toLong())
                                    }
                                    if (pos != b + 1) throw IOException("分段 $i 只收到 ${pos - a} 字节")
                                }
                                ok = true
                                break
                            } catch (e: IOException) {
                                errors[i] = e.message ?: e.toString()
                                Thread.sleep(1000L * attempt)
                            }
                        }
                        if (!ok) break
                        errors.remove(i)
                        finished[i] = true
                        inFile.addAndGet(b + 1 - a)
                        save()
                        report(inFile.get())
                    }
                }
            }.apply { start() }
        }
        workers.forEach { it.join() }
        if (cancelled.get()) return "已停止（下次接着下）"
        if (!finished.all { it }) return "${dest.name}: ${errors.values.firstOrNull() ?: "有分段没下完"}（下次接着下）"
        state.delete()
        return if (part.renameTo(dest)) null else "${dest.name}: 改名失败"
    }

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
