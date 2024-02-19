# Lines that begin with a # are comments
# set the minimum required version of cmake, usually the first line
cmake_minimum_required(VERSION 3.22)

set(CMAKE_C_COMPILER "/usr/bin/clang-17")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-17")
set(CMAKE_CXX_FLAGS "-O3")

# project_name sets the name of the project and causes cmake to
# find the c and c++ compilers
project(reconstruction)

# Find your dependencies.
# Many libraries ship with files that allow CMake to find them
# Then general behavior is to call "find_package" but the options
# provided are package specific.  Usually there is then a CMAKE variable
# That is defined to reference the library
# here: we find the eigen library as per the instruction
# https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(DBoW2 REQUIRED)

# Create an executable from the following source code files
# The Name of the executable creates a cmake "target"
add_executable(reconstruct 
                src/reconstruct.cpp 
                src/features.cpp 
                src/map.cpp
                src/mappoint.cpp
                src/keyframe.cpp
                src/recording.cpp
                src/pinhole.cpp
                src/sfm.cpp
                src/place_recognition.cpp)

add_executable(build_vocab src/build_vocabulary.cpp
                           src/recording.cpp)

add_library(reconstruction src/reconstruct.cpp 
                            src/features.cpp 
                            src/map.cpp
                            src/mappoint.cpp
                            src/keyframe.cpp
                            src/recording.cpp
                            src/pinhole.cpp
                            src/sfm.cpp
                            src/place_recognition.cpp)

# Enable c++17 support.
# Public causes the features to propagate to anything
target_compile_features(reconstruct PUBLIC cxx_std_23)
target_compile_features(reconstruction PUBLIC cxx_std_23)
target_compile_features(build_vocab PUBLIC cxx_std_23)


# Use target_link_libraries to add dependencies to a "target"
# (e.g., a library or executable)
# This will automatically add all required library files
# that need to be linked
# and paths to th locations of header files
target_link_libraries(reconstruct Eigen3::Eigen ${OpenCV_LIBS} Ceres::ceres ${DBoW2_LIBS})
target_link_libraries(reconstruction Eigen3::Eigen ${OpenCV_LIBS} Ceres::ceres ${DBoW2_LIBS})
target_link_libraries(build_vocab  ${OpenCV_LIBS} ${DBoW2_LIBS})

target_include_directories(reconstruct
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>)

target_include_directories(reconstruction
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>)

target_include_directories(build_vocab
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>)

install(DIRECTORY include/reconstruction DESTINATION include)

install(TARGETS reconstruction EXPORT reconstruction-targets)

install(EXPORT reconstruction-targets
        FILE reconstruction-config.cmake
        NAMESPACE reconstruction::
        DESTINATION lib/cmake/${PROJECT_NAME})