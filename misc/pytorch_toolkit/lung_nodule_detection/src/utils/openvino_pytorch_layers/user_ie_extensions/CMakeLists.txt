# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:extension]
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "user_cpu_extension")

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)
# find_package(TBB REQUIRED tbb tbbmalloc)

file(GLOB_RECURSE SRC *.cpp)

add_library(${TARGET_NAME} SHARED ${SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_EXTENSION_API)

if (OpenCV_FOUND)
  target_include_directories(${TARGET_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime
                                            #  ${TBB_IMPORTED_TARGETS}
                                          )
# [cmake:extension]
