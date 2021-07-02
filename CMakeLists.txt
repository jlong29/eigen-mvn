cmake_minimum_required(VERSION 3.1)
project(eigen_mvn CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# the `pkg_check_modules` function is created with this call
find_package(PkgConfig REQUIRED)

# SYSTEM LIBRARIES
pkg_check_modules(GTEST REQUIRED gtest)
if(${GTEST_FOUND})
	 MESSAGE(STATUS "Found GTEST")
	 MESSAGE(STATUS "GTEST INCLUDES: ${GTEST_INCLUDE_DIRS}")
	 MESSAGE(STATUS "GTEST LIBRARIES: ${GTEST_LIBRARIES}")
else()
	 MESSAGE(STATUS "Could not locate GTEST.")
endif()

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
		${GTEST_INCLUDE_DIRS}
)

target_link_libraries(eigen_mvn ${GTEST_LIBRARIES})

# GTests
add_executable(eigen_mvn_test eigen_mvn_test.cpp)
set_target_properties(eigen_mvn_test PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(eigen_mvn_test PRIVATE -O3 -Wall -Wextra)
target_include_directories(
	eigen_mvn_test
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
		$ENV{EIGEN_ROOT_DIR}
		${GTEST_INCLUDE_DIRS}
)
target_link_libraries(eigen_mvn_test eigen_mvn ${GTEST_LIBRARIES})
