cmake_minimum_required(VERSION 3.20)
project(TurbinatorC VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TURBINATOR_ENABLE_CUDA "Build optional CUDA C feature kernels when a CUDA compiler is available" ON)
option(TURBINATOR_BUILD_TESTS "Build native C tests" ON)
option(TURBINATOR_BUILD_TRAINER "Build native C training and monitoring CLI" ON)
option(TURBINATOR_BUILD_VIEWER "Build the native C desktop viewer on Windows" ON)

if(TURBINATOR_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(TURBINATOR_HAS_CUDA ON)
  else()
    message(STATUS "CUDA compiler unavailable; building the CPU C implementation")
  endif()
endif()

add_library(turbinator_sim SHARED
  src/sim/physics.c
  src/sim/sensors.c
  src/sim/c_api.c
  src/trainer/tui.c
)
target_include_directories(turbinator_sim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(turbinator_sim PRIVATE TURBINATOR_SIM_BUILD)

add_library(turbinator_training STATIC src/training/training.c)
target_include_directories(turbinator_training PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(TURBINATOR_HAS_CUDA)
  add_library(turbinator_cuda STATIC src/cuda/features.cu)
  target_include_directories(turbinator_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(turbinator_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(turbinator_cuda PUBLIC TURBINATOR_HAS_CUDA=1)
endif()

if(MSVC)
  target_compile_options(turbinator_sim PRIVATE /W3)
  target_compile_options(turbinator_training PRIVATE /W3)
else()
  target_compile_options(turbinator_sim PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(turbinator_training PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(turbinator_sim PRIVATE m)
  target_link_libraries(turbinator_training PUBLIC m)
endif()

if(WIN32)
  target_link_libraries(turbinator_sim PRIVATE ws2_32)
endif()

if(ANDROID)
  add_library(turbinator_mobile SHARED src/android/native_activity.c)
  target_include_directories(turbinator_mobile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(turbinator_mobile PRIVATE turbinator_sim android log m)
  target_link_options(turbinator_sim PRIVATE "-Wl,-z,max-page-size=16384")
  target_link_options(turbinator_mobile PRIVATE "-Wl,-z,max-page-size=16384")
endif()

if(TURBINATOR_BUILD_TRAINER)
  add_executable(turbinator_trainer src/trainer/main.c src/trainer/http.c)
  target_include_directories(turbinator_trainer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(turbinator_trainer PRIVATE turbinator_sim turbinator_training)
  if(WIN32)
    target_link_libraries(turbinator_trainer PRIVATE ws2_32)
  endif()
  if(TURBINATOR_HAS_CUDA)
    target_link_libraries(turbinator_trainer PRIVATE turbinator_cuda)
  endif()
  add_custom_command(TARGET turbinator_trainer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:turbinator_trainer>
      $<TARGET_FILE_DIR:turbinator_trainer>/turbinator_tui$<TARGET_FILE_SUFFIX:turbinator_trainer>)
endif()

if(WIN32 AND TURBINATOR_BUILD_VIEWER)
  add_executable(turbinator_viewer WIN32 src/viewer/viewer_win32.c)
  target_link_libraries(turbinator_viewer PRIVATE user32 gdi32)
endif()

if(TURBINATOR_BUILD_TESTS)
  include(CTest)
  add_executable(turbinator_tests tests/test_native.c)
  target_link_libraries(turbinator_tests PRIVATE turbinator_sim turbinator_training)
  add_test(NAME native_smoke COMMAND turbinator_tests)
  add_executable(turbinator_training_tests tests/test_training.c)
  target_link_libraries(turbinator_training_tests PRIVATE turbinator_training)
  add_test(NAME training_sac COMMAND turbinator_training_tests)
endif()
