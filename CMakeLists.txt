
# cmake 版本要求
cmake_minimum_required (VERSION 2.6)

# 项目信息
project (Audio CXX CUDA)

# 添加cuda库
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")


# 添加子目录
add_subdirectory("${PROJECT_SOURCE_DIR}/CXYNN")
add_subdirectory("${PROJECT_SOURCE_DIR}/compute-mfcc")

# 添加可执行文件
add_executable (Audio Train.cpp)
target_link_libraries(Audio CXYNeuronNetwork)
target_link_libraries(Audio mfcc)
target_link_libraries(Audio stdc++fs)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-lstdc++fs ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-ggdb ${CMAKE_CXX_FLAGS}")

# 安装选项
set (CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
install (
	TARGETS Audio
	RUNTIME DESTINATION debug
)
