cmake_minimum_required(VERSION 2.8)

PROJECT(opencltest)

IF(APPLE)
  include_directories(/usr/local/include)
  link_directories(/usr/local/lib)
  set(CMAKE_CXX_FLAGS "-g -Wall")
  set(CMAKE_C_FLAGS "-g -Wall")
  FIND_LIBRARY(OPENCL_LIBRARY OpenCL)
  SET(EXTRA_LIBS ${OPENCL_LIBRARY})
ENDIF(APPLE)

add_executable(opencltest main.cpp)
target_link_libraries(opencltest ${EXTRA_LIBS})

INSTALL(TARGETS opencltest RUNTIME DESTINATION bin)
