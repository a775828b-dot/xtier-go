package io.github.a775828b.xtiergo

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Explicit opt-in integration test; mirror must serve the pinned model card's files. */
@RunWith(AndroidJUnit4::class)
class ModelDownloadTest {
    @Test fun downloadAndVerifyPinnedCheckpoint() {
        val mirror = InstrumentationRegistry.getArguments().getString("modelMirror").orEmpty()
        assumeTrue("Pass modelMirror to download the real checkpoint", mirror.isNotEmpty())
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val card = Card.loadAll(ctx).first()
        val store = ModelStore(ctx)
        val error = Downloader(store).run(card, mirror) {}
        assertNull(error, error)
        assertTrue("Checkpoint must be verified before Ready", store.ready(card))
    }
}
