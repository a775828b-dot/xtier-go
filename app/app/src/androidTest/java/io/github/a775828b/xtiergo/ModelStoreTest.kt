package io.github.a775828b.xtiergo

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.Assert.*
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class ModelStoreTest {
    private val abcHash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    private lateinit var store: ModelStore
    private lateinit var card: Card

    @Before fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        store = ModelStore(context)
        card = Card.loadAll(context).first().copy(
            id = "verification-test", dir = "__verification_test_${System.nanoTime()}",
            files = listOf(ModelFile("test.bin", 3, abcHash)),
        )
        store.dir(card).mkdirs()
    }

    @After fun tearDown() {
        File(store.dir(card), "test.bin").delete()
        store.dir(card).delete()
    }

    @Test fun testSameSizeFileCannotStartBeforeVerification() {
        File(store.dir(card), "test.bin").writeText("abc")
        assertEquals(ModelStore.FileState.Unverified, store.status(card).single().state)
        assertFalse(store.ready(card))
        assertTrue(store.verify(card).isEmpty())
        assertTrue(store.ready(card))
    }

    @Test fun testSameSizeCorruptionIsRejected() {
        File(store.dir(card), "test.bin").writeText("abd")
        assertEquals(listOf("test.bin"), store.verify(card))
        assertEquals(ModelStore.FileState.Bad, store.status(card).single().state)
        assertFalse(store.ready(card))
    }

    @Test fun testMissingFileDoesNotReportVerificationSuccess() {
        assertEquals(listOf("test.bin"), store.verify(card))
        assertFalse(store.ready(card))
    }

    @Test fun testDownloaderVerifiesExistingFileWithoutNetwork() {
        File(store.dir(card), "test.bin").writeText("abc")
        assertFalse(store.ready(card))
        assertNull(Downloader(store).run(card, "http://127.0.0.1:1") {})
        assertTrue(store.ready(card))
    }
}
