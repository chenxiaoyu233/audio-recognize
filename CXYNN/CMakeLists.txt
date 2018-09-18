# 选项
option (ENABLE_CUDA "enable the cuda" OFF)
option (NDEBUG "disable the assert" OFF)

# cmake 控制的配置头文件
configure_file (
	"${PROJECT_SOURCE_DIR}/CXYNN/config.h.in"
	"${PROJECT_SOURCE_DIR}/CXYNN/config.h"
)

# 添加头文件搜索目录
include_directories ( "${PROJECT_SOURCE_DIR}/CXYNN" )

# 搜集根目录下的所有源文件
aux_source_directory (. source_list)

# 添加优化/调试编译指令
if (NDEBUG)
	set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")
else (NDEBUG)
	set(CMAKE_CXX_FLAGS "-ggdb ${CMAKE_CXX_FLAGS}")
endif (NDEBUG)

# 编译目标
add_library (CXYNeuronNetwork ${source_list})

# 添加cuda库
if (ENABLE_CUDA)
	include_directories( 
		"${PROJECT_SOURCE_DIR}/CXYNN/cuda" 
	)
	add_subdirectory (cuda)
	target_link_libraries (CXYNeuronNetwork cuda_kernel)
	if (NOT NDEBUG)
		set (CUDA_FLAGS "-g ${CUDA_FLAGS}")
		set (CUDA_FLAGS "-G ${CUDA_FLAGS}")
		set (CUDA_FLAGS "-lineinfo ${CUDA_FLAGS}")
	endif (NOT NDEBUG)
	set (CUDA_FLAGS "-arch sm_61 ${CUDA_FLAGS}")
endif (ENABLE_CUDA)

