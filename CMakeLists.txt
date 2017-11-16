cmake_minimum_required(VERSION 3.4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++0x")

PROJECT(ArcZeeMath)
SET(PROJECT_SRCROOT_PATH "${CMAKE_SOURCE_DIR}/src/")      #工程的根目录，即ArcZeeMath

SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")    #可执行生成后存放的目录(CMAKE_SOURCE_DIR是cmake目录)
SET(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/lib/")       #静态库生成后存放的目录


SET(OPENCL_INCLUDE_PATH /usr/include CACHE PATH "Path to OpenCL include files")
SET(OPENCL_LIB_PATH /usr/lib CACHE PATH "Path to OpenCL libraries")
include_directories(${OPENCL_INCLUDE_PATH})  #指定OpenCL头文件的搜索目录
link_directories(${OPENCL_LIB_PATH})  #指定OpenCL库的搜索目录


INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/include/")   #告诉CMake头文件在哪里？
LINK_DIRECTORIES("${CMAKE_SOURCE_DIR}/lib")              #告诉CMake静态库在哪里？

ADD_SUBDIRECTORY(src)                   #多目录，把src目录加进来，src里面才是真正编译main和hello的
