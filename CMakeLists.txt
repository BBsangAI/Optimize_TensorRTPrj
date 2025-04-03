# 指定CMake的最低版本
cmake_minimum_required(VERSION 3.11)
# 定义项目名称和语言
project(Opt_CabinPrj  LANGUAGES CXX CUDA)

# 设置 C++ 标准为 C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)  # 启用自动 MOC
set(CMAKE_AUTOUIC ON)   # 启用自动 UIC (如果你有 .ui 文件)
set(CMAKE_AUTORCC ON)   # 启用自动 RCC (如果你有 .qrc 文件)
# 查找所需要的包
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)
# 指定要编译的源文件
# file(GLOB SOURCES  src/*.cpp OptmiziePrj/src/*.cpp QT/*.cpp)
file(GLOB SOURCES  OptmiziePrj/src/*.cpp  QT/*.cpp)
add_executable(main ${SOURCES})    # 格式：(可执行文件 要编译的文件)

# 添加库到可执行文件
target_link_libraries(main ${OpenCV_LIBS} 
                        Threads::Threads 
                        nvonnxparser 
                        nvinfer 
                        nvinfer_plugin
                        rt 
                        ${CUDA_LIBRARIES}
                        Qt5::Widgets
                        /usr/local/cuda/lib64/libcudart.so              # cuda runtime
                        #/home/jetson/Desktop/Opt_CabinPrj/OptmiziePrj/src/trt_proprecess.o
                        stdc++fs
                        ) 
# 添加头文件到可执行文件
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS}                          # opencv
                                        /home/jetson/Desktop/Opt_CabinPrj/QT            # QT
                                        /usr/include/aarch64-linux-gnu/                  # tensorrt
                                        /home/jetson/Desktop/Opt_CabinPrj/OptmiziePrj/include
                                        /usr/local/cuda/include)                        # cuda
