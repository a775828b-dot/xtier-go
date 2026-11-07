package io.github.a775828b.xtiergo

/** The engine's C API (server/xtg_capi.h) through app/src/main/cpp/jni.cpp. One server per process. */
object Native {
    init {
        System.loadLibrary("xtiergo")
    }

    /** Starts loading and serving in the background; null when it started, else why not. */
    external fun start(configJson: String): String?

    external fun stop()

    /** The same JSON as GET /xtg/status. */
    external fun status(): String

    /** Memory the options would take for this model directory (JSON), without loading the weights. */
    external fun estimate(modelDir: String, optionsJson: String): String

    external fun version(): String
}
