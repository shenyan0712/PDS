cmake_minimum_required(VERSION 3.4)

FILE(GLOB SOURCE_1 "${PROJECT_SRCROOT_PATH}/demo/autodiff_test.cpp")     	#告诉CMake源文件在哪里？	
ADD_EXECUTABLE(autodiff_test ${SOURCE_1})                          		#告诉CMake生成一个main可执行文件
TARGET_LINK_LIBRARIES(autodiff_test ArcZeeMath)                         #告诉CMake静态库在哪里？


FILE(GLOB SOURCE_2 "${PROJECT_SRCROOT_PATH}/demo/chol_test.cpp")     	#告诉CMake源文件在哪里？
ADD_EXECUTABLE(chol_test ${SOURCE_2})                          		#告诉CMake生成一个main可执行文件
TARGET_LINK_LIBRARIES(chol_test ArcZeeMath)                         #告诉CMake静态库在哪里？