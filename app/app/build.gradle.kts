plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// -PxtgQnnLibs=DIR: also pack DIR/arm64-v8a (the QNN SDK's HTP libraries, and qnn-net-run / qnn-profile-viewer
// renamed libqnn_net_run.so / libqnn_profile_viewer.so) for the NPU check in DEV, with the libraries extracted on
// install so those two can run. A local build only: the SDK is not in this repository.
val qnnLibs: String? = providers.gradleProperty("xtgQnnLibs").orNull

android {
    namespace = "io.github.a775828b.xtiergo"
    compileSdk = 35
    ndkVersion = providers.gradleProperty("xtgNdkVersion").getOrElse("27.2.12479018")

    defaultConfig {
        applicationId = "io.github.a775828b.xtiergo"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.0.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // -PxtgAppIdSuffix=.dev: a development build next to the installed app (other signing key, own data)
        val suffix = providers.gradleProperty("xtgAppIdSuffix").getOrElse("")
        applicationIdSuffix = suffix.ifEmpty { null }
        manifestPlaceholders["appLabel"] = if (suffix.isEmpty()) "XTier Go" else "XTier Go$suffix"
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_shared", "-DXTG_TESTS=OFF", "-DXTG_TOOLS=OFF")
                cppFlags += listOf("-O3")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // the model cards come straight from configs/ (one source for the cards)
    sourceSets["main"].assets.srcDirs("../../configs")
    if (qnnLibs != null) sourceSets["main"].jniLibs.srcDir(qnnLibs)

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
    packaging { jniLibs { useLegacyPackaging = qnnLibs != null } }
}

dependencies {
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material3:material3-window-size-class")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    // model downloads with our own resolver (DirectDns), ranges over several connections
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
