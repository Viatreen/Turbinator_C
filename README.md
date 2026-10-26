# Turbinator C

Turbinator C is a C11 flight simulation and reinforcement learning project for the TURBINATOR quad-jet VTOL aircraft and TALON-6061 blended-wing-body aircraft. CMake builds the simulator, a native SAC trainer, tests, a Windows telemetry viewer, and an Android NativeActivity library. An optional CUDA C target provides batched sensor feature extraction when a CUDA compiler is available.

## Build

```sh
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

On Windows, the desktop viewer is built as `turbinator_viewer.exe`. The simulator and trainer build without CUDA. Set `-DTURBINATOR_ENABLE_CUDA=OFF` to skip CUDA compiler detection.

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
- **Learning:** compact multimodal Soft Actor-Critic with a replay buffer, twin critics, entropy tuning, reference-controller guidance, and binary checkpoints. The implemented learner uses the CPU; the CUDA C target currently provides feature extraction kernels.
- **Viewer:** Win32/GDI telemetry and replay catalog with separate aircraft views. Wind display starts off.
- **Android:** C NativeActivity with local simulator display, separate BWB and quad views, and live training status polling. Build the Android libraries with the NDK CMake toolchain using `-DTURBINATOR_ENABLE_CUDA=OFF -DTURBINATOR_BUILD_TESTS=OFF -DTURBINATOR_BUILD_TRAINER=OFF`; package both native libraries with `src/android/AndroidManifest.xml`.
- **Tests:** scenario, ABI, sensor, serialization, and learner checks run through CTest.

The executable uses its own compact network and checkpoint format. Large transformer training, distributed GPU optimization, APK packaging, and HDF5 replay playback are outside the current executable feature set. See [the training architecture report](docs/TRAINING_ARCHITECTURE.md) for the experimental 618M design and prioritized training improvements.

The Android display defaults to `10.0.2.2:8765` for an emulator. For a physical phone, set its private `server_host.txt` to the computer's LAN address and start the trainer with `--serve --serve-host 0.0.0.0`.
