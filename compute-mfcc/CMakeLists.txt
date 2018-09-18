
aux_source_directory (. mfcc_source_list)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

add_library(mfcc ${mfcc_source_list})
