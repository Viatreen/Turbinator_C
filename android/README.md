# Android APK

The Android app uses a C NativeActivity. Gradle packages the native libraries produced by the root CMake project; there are no Java or Kotlin application sources.

Requirements: JDK 17, Android SDK Platform 35, Build Tools 35.0.0, CMake 3.22.1, and NDK 27.2.12479018. Set `JAVA_HOME` and `ANDROID_HOME` to those installations.

From this directory:

```sh
./gradlew :app:assembleDebug
```

On Windows, run `gradlew.bat :app:assembleDebug`. The installable debug APK is `app/build/outputs/apk/debug/app-debug.apk`. It contains arm64 and x86-64 simulator and NativeActivity libraries and targets Android 15 while supporting Android 10 and later. Debug builds use Android's standard local debug signing key.

The app shows a local aircraft simulation and reads training status from the C trainer's `/api/v1/status` endpoint. The emulator defaults to `10.0.2.2:8765`; a physical phone can use the computer's LAN address through the app's private `server_host.txt` setting. Start the trainer with `--serve --serve-host 0.0.0.0` for access from another device on the same network.
