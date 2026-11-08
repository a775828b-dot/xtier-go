package io.github.a775828b.xtiergo

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * A model card: configs/<model>-<device>/card.json, the same fields as XTier's configs/<model>/console.py
 * (options with defaults = the measured configuration, choices, off_when, unavailable, notes) plus the files to
 * download and where each option goes: a pack option, a chat_template_kwargs default, or a launch setting.
 */
data class Choice(val value: String, val label: String)

data class Option(
    val key: String,
    val label: String,
    val default: String,
    val choices: List<Choice>,
    val hint: String,
    val target: String,                       // option | kwarg | launch
    val kwarg: String?,
    val map: JSONObject?,                     // choice -> JSON value for a kwarg
    val offWhen: List<Map<String, String>>,
    val unavailable: Map<String, String>,
    val switch: Boolean,
)

data class Note(val whenAll: Map<String, String>, val text: String, val warn: Boolean)

// whenAll: needed only under these option values (one context bucket's .pte); local: no download source (adb push)
data class ModelFile(
    val name: String,
    val size: Long,
    val sha256: String?,
    val whenAll: Map<String, String> = emptyMap(),
    val local: Boolean = false,
)

data class Card(
    val id: String,
    val order: Int,
    val name: String,
    val code: String,
    val port: Int,
    val served: String,
    val arch: String,
    val hardware: String,
    val readyTimeout: Int,
    val repo: String,
    val revision: String,
    val dir: String,
    val mirrors: List<String>,
    val files: List<ModelFile>,
    val benchKind: String,
    val benchDec: String,
    val benchPre: String,
    val benchTitle: String,
    val options: List<Option>,
    val notes: List<Note>,
) {
    fun defaults(): Map<String, String> = options.associate { it.key to it.default }

    /** Saved values that are still valid, the rest at their defaults. */
    fun resolve(saved: Map<String, String>): Map<String, String> =
        options.associate { o ->
            val v = saved[o.key]
            o.key to (if (v != null && o.choices.any { it.value == v } && v !in o.unavailable) v else o.default)
        }

    fun isOff(o: Option, values: Map<String, String>): Boolean =
        o.offWhen.any { cond -> cond.all { (k, v) -> values[k] == v } }

    fun activeNotes(values: Map<String, String>): List<Note> =
        notes.filter { n -> n.whenAll.all { (k, v) -> values[k] == v } }

    /** The files these option values need. */
    fun filesFor(values: Map<String, String>): List<ModelFile> =
        files.filter { f -> f.whenAll.all { (k, v) -> values[k] == v } }

    /** The pack options, the template kwargs and the launch settings for these values. */
    fun launchConfig(modelDir: String, values: Map<String, String>): JSONObject {
        val options = JSONObject()
        val kwargs = JSONObject()
        var threads = 0
        var specK = 0
        for (o in this.options) {
            val v = values[o.key] ?: o.default
            when (o.target) {
                "option" -> options.put(o.key, v)
                "kwarg" -> kwargs.put(o.kwarg ?: o.key, o.map?.opt(v) ?: v)
                "launch" -> when (o.key) {
                    "threads" -> threads = v.toIntOrNull() ?: 0
                    "spec_k" -> specK = v.toIntOrNull() ?: 0
                }
            }
        }
        return JSONObject()
            .put("model_dir", modelDir)
            .put("host", "127.0.0.1")
            .put("port", port)
            .put("served", served)
            .put("threads", threads)
            .put("spec_k", specK)
            .put("options", options)
            .put("template_kwargs", kwargs)
    }

    /** Pack options only (for the memory estimate). */
    fun packOptions(values: Map<String, String>): JSONObject {
        val o = JSONObject()
        for (opt in options) if (opt.target == "option") o.put(opt.key, values[opt.key] ?: opt.default)
        return o
    }

    companion object {
        private fun strings(o: JSONObject?): Map<String, String> {
            if (o == null) return emptyMap()
            return o.keys().asSequence().associateWith { o.get(it).toString() }
        }

        fun parse(j: JSONObject): Card {
            val model = j.getJSONObject("model")
            val files = model.getJSONArray("files").let { a ->
                (0 until a.length()).map { i ->
                    val f = a.getJSONObject(i)
                    ModelFile(f.getString("name"), f.getLong("size"), f.optString("sha256").ifEmpty { null },
                        strings(f.optJSONObject("when")), f.optBoolean("local", false))
                }
            }
            val mirrors = model.optJSONArray("mirrors")?.let { a -> (0 until a.length()).map { a.getString(it) } }
                ?: listOf("https://huggingface.co")
            val options = j.getJSONArray("options").let { a ->
                (0 until a.length()).map { i ->
                    val o = a.getJSONObject(i)
                    val choices = o.getJSONArray("choices").let { c ->
                        (0 until c.length()).map { k -> c.getJSONArray(k).let { Choice(it.getString(0), it.getString(1)) } }
                    }
                    val offWhen = o.optJSONArray("off_when")?.let { c -> (0 until c.length()).map { strings(c.getJSONObject(it)) } }
                        ?: emptyList()
                    Option(
                        key = o.getString("key"),
                        label = o.getString("label"),
                        default = o.getString("default"),
                        choices = choices,
                        hint = o.optString("hint"),
                        target = o.optString("target", "option"),
                        kwarg = o.optString("kwarg").ifEmpty { null },
                        map = o.optJSONObject("map"),
                        offWhen = offWhen,
                        unavailable = strings(o.optJSONObject("unavailable")),
                        switch = o.optBoolean("switch", false),
                    )
                }
            }
            val notes = j.optJSONArray("notes")?.let { a ->
                (0 until a.length()).map { i ->
                    val n = a.getJSONObject(i)
                    Note(strings(n.optJSONObject("when")), n.getString("text"), n.optBoolean("warn", false))
                }
            } ?: emptyList()
            val bench = j.optJSONObject("bench") ?: JSONObject()
            return Card(
                id = j.getString("id"), order = j.optInt("order", 100), name = j.getString("name"),
                code = j.optString("code"), port = j.getInt("port"), served = j.getString("served"),
                arch = j.optString("arch"), hardware = j.optString("hardware"),
                readyTimeout = j.optInt("ready_timeout", 600),
                repo = model.getString("repo"), revision = model.optString("revision", "main"),
                dir = model.getString("dir"), mirrors = mirrors, files = files,
                benchKind = bench.optString("kind", "agent"), benchDec = bench.optString("dec", "--"),
                benchPre = bench.optString("pre", "--"), benchTitle = bench.optString("title"),
                options = options, notes = notes,
            )
        }

        /** Every assets/<dir>/card.json (copied from configs/ at build time), by order. */
        fun loadAll(context: Context): List<Card> {
            val assets = context.assets
            val out = mutableListOf<Card>()
            for (dir in assets.list("")?.toList() ?: emptyList()) {
                val names = assets.list(dir)?.toList() ?: continue
                if ("card.json" !in names) continue
                val text = assets.open("$dir/card.json").bufferedReader().use { it.readText() }
                out += parse(JSONObject(text))
            }
            return out.sortedBy { it.order }
        }

        fun values(json: String?): Map<String, String> =
            if (json.isNullOrEmpty()) emptyMap() else strings(JSONObject(json))

        fun valuesJson(values: Map<String, String>): String = JSONObject(values as Map<*, *>).toString()

        @Suppress("unused")
        fun array(a: JSONArray?): List<String> = a?.let { (0 until it.length()).map(it::getString) } ?: emptyList()
    }
}
