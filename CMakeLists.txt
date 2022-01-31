cmake_minimum_required(VERSION 3.1)
project(eigen-mvn CXX)

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

add_executable(eigen_mvn_test eigen_mvn_test.cpp)
set_target_properties(eigen_mvn_test PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(eigen_mvn_test PRIVATE -O3 -Wall -Wextra)
target_include_directories(
	eigen_mvn_test
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
		$ENV{EIGEN_ROOT_DIR}
)
target_link_libraries(eigen_mvn_test eigen_mvn)

install(TARGETS eigen_mvn DESTINATION ${CMAKE_CURRENT_BINARY_DIR})