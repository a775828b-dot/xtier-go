package io.github.a775828b.xtiergo

import android.content.Context
import java.io.File
import java.io.FileInputStream
import java.util.concurrent.TimeUnit

/**
 * The NPU check from the app's own process. adb's run-as gives the runas_app SELinux domain, so only the app can
 * tell whether untrusted_app reaches the CDSP: this runs the QNN SDK's qnn-net-run on every context binary in
 * <external files>/npu/ (inputs and list_m<M>.txt beside them, as in tools/npu_probe's kit) and writes
 * npu/selftest.txt and the first outputs under npu/out/. Only a build with -PxtgQnnLibs=DIR carries the tools.
 * From adb: am start -n <package>/io.github.a775828b.xtiergo.MainActivity --ez npu_selftest true
 */
object NpuProbe {
    private fun lib(c: Context) = File(c.applicationInfo.nativeLibraryDir)

    fun available(c: Context) = File(lib(c), "libqnn_net_run.so").exists()

    fun dir(c: Context) = File(c.getExternalFilesDir(null), "npu")

    /** One line per model ("name rc=0 · 20 次 · NetRun 47.0 ms · NPU 41.3 ms"), after the process facts. */
    fun run(c: Context, inferences: Int = 20): List<String> {
        val lib = lib(c)
        val dir = dir(c).apply { mkdirs() }
        val lines = mutableListOf(
            "进程 uid ${android.os.Process.myUid()} · ${runCatching { File("/proc/self/attr/current").readText().trim('\u0000', ' ', '\n') }.getOrDefault("?")}",
            "/dev/fastrpc-cdsp：${openable("/dev/fastrpc-cdsp")}",
        )
        if (!available(c)) {
            lines += "这个版本没带 QNN 工具（编译时加 -PxtgQnnLibs）"
            return lines
        }
        // an executable under /data/app gets the linker's restricted section: libcdsprpc.so (the HTP stub needs it)
        // is found only with /vendor/lib64 on the path
        val env = mapOf(
            "LD_LIBRARY_PATH" to "${lib.path}:/vendor/lib64",
            "ADSP_LIBRARY_PATH" to "${lib.path};/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp",
        )
        val models = dir.listFiles { f -> f.name.endsWith(".bin") }?.sortedBy { it.name }.orEmpty()
        if (models.isEmpty()) lines += "${dir.path} 里没有 .bin"
        for (m in models) {
            val name = m.name.removeSuffix(".bin")
            val rows = Regex("_m(\\d+)").find(name)?.groupValues?.get(1)
            if (rows == null) {
                lines += "$name：名字里没有 _m<M>，跳过"
                continue
            }
            val out = File(dir, "out/$name").apply { deleteRecursively(); mkdirs() }
            val t0 = System.nanoTime()
            val (rc, log) = exec(
                listOf(
                    File(lib, "libqnn_net_run.so").path, "--backend", File(lib, "libQnnHtp.so").path,
                    "--retrieve_context", m.path, "--input_list", "list_m$rows.txt", "--output_dir", out.path,
                    "--perf_profile", "burst", "--profiling_level", "basic", "--num_inferences", "$inferences",
                    "--keep_num_outputs", "1", "--log_level", "warn",
                ), dir, env,
            )
            val wall = (System.nanoTime() - t0) / 1e9
            // qnn-profiling-data_0.log, or without the _0 when the app cannot make qnn-net-run's symlink
            val prof = listOf("qnn-profiling-data_0.log", "qnn-profiling-data.log").map { File(out, it) }.firstOrNull { it.exists() }
                ?: File(out, "qnn-profiling-data_0.log")
            val avg = if (prof.exists())
                average(exec(listOf(File(lib, "libqnn_profile_viewer.so").path, "--input_log", prof.path), dir, env).second)
            else "没有 profile"
            lines += "$name rc=$rc · $inferences 次 · 墙钟 %.1f s · $avg".format(wall)
            if (rc != 0) lines += log.takeLast(1200).trim()
        }
        File(dir, "selftest.txt").writeText(lines.joinToString("\n") + "\n")
        return lines
    }

    private fun openable(path: String): String =
        try {
            FileInputStream(path).close(); "能打开"
        } catch (e: Exception) {
            "打不开（${e.message}）"
        }

    private fun exec(cmd: List<String>, cwd: File, env: Map<String, String>): Pair<Int, String> {
        val p = ProcessBuilder(cmd).directory(cwd).redirectErrorStream(true).apply { environment().putAll(env) }.start()
        val text = StringBuilder()
        val reader = Thread { p.inputStream.bufferedReader().forEachLine { text.appendLine(it) } }.apply { start() }
        if (!p.waitFor(10, TimeUnit.MINUTES)) p.destroyForcibly().waitFor()
        reader.join(5000)
        return p.exitValue() to text.toString()
    }

    /** "NetRun 47.0 ms · NPU 41.3 ms" from the Average section of qnn-profile-viewer's output. */
    private fun average(text: String): String {
        val sec = text.substringAfter("Execute Stats (Average)", "").substringBefore("Execute Stats (Min)")
        fun us(key: String) = Regex("$key:\\s*([\\d.]+)\\s*us").find(sec)?.groupValues?.get(1)?.toDouble()
        val net = us("NetRun") ?: return "profile 里没有 Average"
        val npu = us("Accelerator \\(execute\\) time\\)")
        return "NetRun %.1f ms".format(net / 1000) + (npu?.let { " · NPU %.1f ms".format(it / 1000) } ?: "")
    }
}
