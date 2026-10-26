# Turbinator C

Turbinator C is a C11 flight simulation and reinforcement learning project for the TURBINATOR quad-jet VTOL aircraft and TALON-6061 blended-wing-body aircraft. CMake builds the simulator, a native SAC trainer, tests, a Windows telemetry viewer, and an Android NativeActivity library. CUDA C targets provide batched sensor features and a large-network training library when a CUDA compiler is available.

## Build

```sh
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

On Windows, the desktop viewer is built as `turbinator_viewer.exe`. The simulator and compact CPU trainer build without CUDA. Set `-DTURBINATOR_ENABLE_CUDA=OFF` to skip CUDA compiler detection.

## Run

```sh
./build/turbinator_trainer train --season-dir seasons/demo --batch 8 --steps 1000
./build/turbinator_trainer evaluate --checkpoint seasons/demo/checkpoints/ckpt_latest.tbc --trials 4 --steps 1000
./build/turbinator_trainer record --checkpoint seasons/demo/checkpoints/ckpt_latest.tbc --scenario 22 --steps 400 --output replay.jsonl
./build/turbinator_trainer benchmark --batch 8 --steps 100
./build/turbinator_trainer tui --season-dir seasons/demo --serve --serve-port 8765
```

On Windows, use `build\turbinator_trainer.exe` in place of `./build/turbinator_trainer`. The TUI uses `p` to pause, `i` for indefinite training, `c` to save a checkpoint, `e` to request evaluation, `r` to request a replay, `o` to open the desktop viewer, and `q` to stop.

The trainer writes checkpoints, live status, and IPC files beneath the selected season directory. The HTTP listener runs only when `--serve` is supplied. It provides status, basic control, and replay listing endpoints at `/api/v1/`.

## Components

- **Simulator:** C11 batch API with 4,148-value observations, four 64×64 cameras, 4,096 LiDAR bins, 64 privileged critic features, and 15 physical actions. It supports the quad-jet and BWB scenario registry, arena interactions, projectile effects, state serialization, and a BWB aerodynamic lookup table in `data/`.
- **Learning:** compact CPU multimodal Soft Actor-Critic with replay, twin critics, entropy tuning, reference-controller guidance, and binary checkpoints. The optional CUDA C build also defines a 617,973,042-parameter network and GPU training library.
- **Viewer:** Win32/GDI telemetry and replay catalog with separate aircraft views. Wind display starts off.
- **Android:** C NativeActivity with local simulator display, separate BWB and quad views, and live training status polling. The `android/` Gradle project builds its native libraries with the root CMake project and packages an installable APK.
- **Tests:** scenario, ABI, sensor, serialization, and learner checks run through CTest.

The default `turbinator_trainer` executable uses the compact CPU network and its own checkpoint format. HDF5 replay playback is not included. See [the training architecture report](docs/TRAINING_ARCHITECTURE.md) for the 618M design and prioritized training improvements.

## 618M network and GPU requirements

The large-network tensor layout contains **617,973,042 trainable parameters**: 528,338,866 in the actor and 89,634,176 in the twin critics, across 298 tensors. The layout count is checked by a C11 test. A separate research telemetry constant lists a different actor/critic split while retaining the same total; the tensor inventory is the basis for these figures.

The full CUDA C learner targets compute capability 7.0 or newer and uses cuBLAS. Its default CMake CUDA architecture is `70` for Tesla V100. Building it requires a supported CUDA toolkit and compatible CMake and host compiler toolchains. The GPU trainer defaults to two samples and 512 replay transitions per device. It stores FP32 parameters, gradients, target parameters, and RMSprop state, requiring about **9.2 GiB per GPU** for those four model-sized buffers alone. Replay adds 66,672 bytes per transition (about 33 MiB at the default 512-transition capacity or 1.02 GiB at 16,384 transitions); activations, minibatch scratch, CUDA context, and cuBLAS workspace add further memory. A V100 with 16 GiB is a practical starting point, and available device memory must be checked for the chosen batch and replay settings.

The CUDA build passes its CTest suite. A one-V100 smoke run completed two environment steps and two optimizer updates with finite losses. Checkpoint save/resume completed another update, and four V100s completed 32 updates through replica synchronization. A separate full-network test checked finite forward outputs and nonzero gradients across sensor stems, fusion, MoE, action, auxiliary, world, and both critics. These checks establish execution; learned flight performance remains to be measured. See the report's [validation gates](docs/TRAINING_ARCHITECTURE.md#7-suggested-validation-gates).

```sh
cmake -S . -B build-cuda -DTURBINATOR_ENABLE_CUDA=ON -DTURBINATOR_CUDA_ARCHITECTURES=70
cmake --build build-cuda --parallel
./build-cuda/turbinator_trainer network-info
./build-cuda/turbinator_trainer train --engine cuda-618m --gpu-count 1 --gpu-batch 1 --gpu-replay-capacity 8 --batch 1 --steps 2 --no-save
```

The last command is a small execution check and disables checkpoint writing. For a normal run, omit `--no-save`; GPU checkpoints are directories under the selected season and contain model, target, optimizer, and trainer state. Put the CUDA toolkit's binary directory on `PATH` at runtime.

The 618M learner currently uses deterministic rollout actions and an analytic entropy proxy. It uses one-moment RMSprop and saves model, target, optimizer, and trainer metadata; replay starts empty after checkpoint load. It is a SAC-style training implementation. The architecture report lists the remaining training improvements.

## Quadro P3000 6 GB configuration

The full FP32 learner's model state exceeds 6 GB. The `p3000` CMake preset builds a low-memory training path for Pascal `sm_61`: CUDA extracts the 142 sensor features in batches, while the compact actor, critics, replay, and optimizer run in C11. The preset uses a Release build and AVX2 for the CPU learner by default; disable AVX2 with `-DTURBINATOR_P3000_AVX2=OFF` if the host CPU requires it. Use a CUDA 12.x toolkit for Pascal compilation.

```sh
cmake --preset p3000
cmake --build --preset p3000 --parallel
ctest --preset p3000
./build-p3000/turbinator_trainer train --engine cuda-p3000 --season-dir seasons/laptop --steps 1000
```

The P3000 engine defaults to 32 simulator environments, accepts `--gpu-device ID`, and writes compact C11 checkpoints that the regular `evaluate` and `record` commands can load. Its persistent CUDA feature buffers use about 1 MiB at batch 32, plus CUDA context memory. The `sm_61` preset passed its tests and a short training run on an available 2 GB Pascal GPU; throughput on a Quadro P3000 has not been measured. You can also run `--engine cpu` in the same build and compare throughput; the checkpoint format is shared.

The Android display defaults to `10.0.2.2:8765` for an emulator. For a physical phone, set its private `server_host.txt` to the computer's LAN address and start the trainer with `--serve --serve-host 0.0.0.0`.

## Android APK

Install JDK 17, Android SDK Platform 35, Build Tools 35.0.0, CMake 3.22.1, and NDK 27.2.12479018. Set `JAVA_HOME` and `ANDROID_HOME`, then build from `android/`:

```sh
./gradlew :app:assembleDebug
```

On Windows use `gradlew.bat`. The APK is `android/app/build/outputs/apk/debug/app-debug.apk`; it includes arm64 and x86-64 native libraries. The debug APK was installed and launched on an Android 35 emulator, and its signature and 16 KB alignment were checked. See [Android build notes](android/README.md).
