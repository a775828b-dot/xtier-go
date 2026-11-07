package io.github.a775828b.xtiergo

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import java.io.File

/** What the launcher shows and checks about the device: SoC, memory, temperature, battery, CPU features. */
data class DeviceInfo(
    val soc: String,
    val totalBytes: Long,
    val availBytes: Long,
    val lowMemory: Boolean,
    val thermalStatus: Int,           // PowerManager.THERMAL_STATUS_*
    val thermalHeadroom: Float,       // 1.0 = throttling starts; NaN when unknown
    val batteryTempC: Float,
    val batteryPercent: Int,
    val charging: Boolean,
    val batteryWatts: Float,          // + charging, - draining; NaN when unknown
    val cpuFeatures: String,
) {
    companion object {
        fun read(context: Context): DeviceInfo {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
            val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            val headroom = if (Build.VERSION.SDK_INT >= 30) pm.getThermalHeadroom(10) else Float.NaN
            val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val temp = (battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10f
            val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
            val status = battery?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
            val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
            val microAmps = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
            val volts = (battery?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0) / 1000f
            val watts = if (microAmps == Long.MIN_VALUE || volts <= 0f) Float.NaN else microAmps / 1e6f * volts
            val soc = if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE
            return DeviceInfo(
                soc = soc,
                totalBytes = mi.totalMem,
                availBytes = mi.availMem,
                lowMemory = mi.lowMemory,
                thermalStatus = pm.currentThermalStatus,
                thermalHeadroom = headroom,
                batteryTempC = temp,
                batteryPercent = if (level >= 0 && scale > 0) level * 100 / scale else -1,
                charging = status == BatteryManager.BATTERY_STATUS_CHARGING || status == BatteryManager.BATTERY_STATUS_FULL,
                batteryWatts = watts,
                cpuFeatures = cpuFeatures(),
            )
        }

        /** The aarch64 features the kernels care about, from /proc/cpuinfo. */
        private fun cpuFeatures(): String = try {
            val line = File("/proc/cpuinfo").readLines().firstOrNull { it.startsWith("Features") } ?: ""
            val have = line.substringAfter(":").trim().split(" ").toSet()
            listOf("asimddp", "i8mm", "bf16", "sve", "sve2", "sme").filter { it in have }.joinToString(" ")
        } catch (e: Exception) {
            ""
        }

        fun thermalLabel(status: Int): String = when (status) {
            PowerManager.THERMAL_STATUS_NONE -> "正常"
            PowerManager.THERMAL_STATUS_LIGHT -> "轻微"
            PowerManager.THERMAL_STATUS_MODERATE -> "中等"
            PowerManager.THERMAL_STATUS_SEVERE -> "严重"
            PowerManager.THERMAL_STATUS_CRITICAL -> "危急"
            PowerManager.THERMAL_STATUS_EMERGENCY, PowerManager.THERMAL_STATUS_SHUTDOWN -> "关机前"
            else -> "未知"
        }
    }
}

fun gib(bytes: Long): String = "%.1f GiB".format(bytes / 1073741824.0)
