cmake_minimum_required(VERSION 3.22)

set(CMAKE_C_COMPILER "/usr/bin/clang-17")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-17")
set(CMAKE_CXX_FLAGS "-O3")

project(reconstruction)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(DBoW2 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)

add_executable(reconstruct 
                src/reconstruct.cpp 
                src/features.cpp 
                src/map.cpp
                src/mappoint.cpp
                src/keyframe.cpp
                src/recording.cpp
                src/pinhole.cpp
                src/sfm.cpp
                src/loop.cpp
                src/place_recognition.cpp
                src/dense_reconstruction.cpp)

target_compile_features(reconstruct PUBLIC cxx_std_23)

target_link_libraries(reconstruct Eigen3::Eigen ${OpenCV_LIBS} Ceres::ceres ${DBoW2_LIBS} Sophus::Sophus ${Boost_LIBRARIES})

target_include_directories(reconstruct
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>
    ${Boost_INCLUDE_DIR})
