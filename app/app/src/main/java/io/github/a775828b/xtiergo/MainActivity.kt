package io.github.a775828b.xtiergo

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        // the console is always dark: light system-bar icons on a transparent bar, the page under it
        enableEdgeToEdge(SystemBarStyle.dark(Color.TRANSPARENT), SystemBarStyle.dark(Color.TRANSPARENT))
        super.onCreate(savedInstanceState)
        window.decorView.setBackgroundColor(0xFF03050C.toInt())
        // the server's notification (the foreground service runs without it, just invisibly)
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        model = LauncherModel(applicationContext)
        multiWindow = isInMultiWindowMode
        setContent { XTierTheme { Launcher(model, multiWindow) } }
        selfTest(intent)
    }

    // split screen and floating windows show the rate board (Launcher)
    private var multiWindow by mutableStateOf(false)

    override fun onMultiWindowModeChanged(isInMultiWindowMode: Boolean, newConfig: Configuration) {
        super.onMultiWindowModeChanged(isInMultiWindowMode, newConfig)
        multiWindow = isInMultiWindowMode
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        selfTest(intent)
    }

    private lateinit var model: LauncherModel

    // adb: am start -n <package>/io.github.a775828b.xtiergo.MainActivity --ez npu_selftest true
    private fun selfTest(intent: Intent?) {
        if (intent?.getBooleanExtra("npu_selftest", false) == true) lifecycleScope.launch { model.npuSelfTest() }
    }
}
