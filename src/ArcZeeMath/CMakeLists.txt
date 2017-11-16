FILE(GLOB SOURCE_1 "${PROJECT_SRCROOT_PATH}/ArcZeeMath/*.cpp")      #告诉CMake源文件在哪里？
ADD_LIBRARY(ArcZeeMath STATIC ${SOURCE_1})                       #告诉CMake生成的是一个静态库
TARGET_LINK_LIBRARIES(ArcZeeMath OpenCL)  #实际会形成OpenCL.lib库依赖