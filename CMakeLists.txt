cmake_minimum_required(VERSION 3.16)
project(wsi_scan_pipeline CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(WSI_WITH_ONNX "Build the ONNX Runtime tile-quality classifier" OFF)
set(ORT_ROOT "" CACHE PATH "Root of an unpacked onnxruntime release (has include/ and lib/)")

add_library(wsi_core STATIC
  src/image.cpp
  src/focus.cpp
  src/segment.cpp
  src/registration.cpp
  src/tiff.cpp
  src/camera.cpp
  src/slide.cpp
  src/scan.cpp
  src/classifier.cpp
)
target_include_directories(wsi_core PUBLIC include)

if(MSVC)
  target_compile_options(wsi_core PUBLIC /W4 /Zc:__cplusplus /permissive-)
else()
  target_compile_options(wsi_core PUBLIC -Wall -Wextra)
endif()

if(WSI_WITH_ONNX)
  if(NOT ORT_ROOT)
    message(FATAL_ERROR "WSI_WITH_ONNX=ON requires -DORT_ROOT=<unpacked onnxruntime release>")
  endif()
  target_compile_definitions(wsi_core PUBLIC WSI_WITH_ONNX=1)
  target_include_directories(wsi_core PUBLIC "${ORT_ROOT}/include")
  find_library(ORT_LIB onnxruntime HINTS "${ORT_ROOT}/lib" REQUIRED)
  target_link_libraries(wsi_core PUBLIC "${ORT_LIB}")
endif()

add_executable(wsi_makeslide apps/makeslide.cpp)
target_link_libraries(wsi_makeslide PRIVATE wsi_core)

add_executable(wsi_scan apps/scan_main.cpp)
target_link_libraries(wsi_scan PRIVATE wsi_core)

add_executable(wsi_bench apps/bench.cpp)
target_link_libraries(wsi_bench PRIVATE wsi_core)

add_executable(wsi_test tests/test_main.cpp)
target_link_libraries(wsi_test PRIVATE wsi_core)

enable_testing()
add_test(NAME unit COMMAND wsi_test)

if(WSI_WITH_ONNX AND WIN32)
  # The runtime DLL has to sit next to the binaries that load it.
  add_custom_command(TARGET wsi_scan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:wsi_scan>")
endif()
