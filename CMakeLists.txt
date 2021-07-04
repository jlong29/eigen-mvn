cmake_minimum_required(VERSION 3.1)
project(eigen_mvn CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LOCAL LIBRARIES
add_library(eigen_mvn
		eigen_mvn.h
		eigen_mvn.cpp
)

target_include_directories(
	eigen_mvn
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
		$ENV{EIGEN_ROOT_DIR}
)

target_link_libraries(eigen_mvn ${GTEST_LIBRARIES})
