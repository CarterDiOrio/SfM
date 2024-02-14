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
find_package(glfw3 REQUIRED)

add_library(imgui
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imconfig.h
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_internal.h
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imstb_rectpack.h
    ${imgui_SOURCE_DIR}/imstb_textedit.h
    ${imgui_SOURCE_DIR}/imstb_truetype.h
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.h
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.h
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw)

add_library(glad
    ${glad_SOURCE_DIR}/src/glad.c
)
target_include_directories(glad PUBLIC ${glad_SOURCE_DIR}/include)

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
                src/bundle_adjust.cpp)

add_executable(visualizer
                src/visualizer.cpp)


# Enable c++17 support.
# Public causes the features to propagate to anything
target_compile_features(reconstruct PUBLIC cxx_std_23)
target_compile_features(visualizer PUBLIC cxx_std_23)

# Use target_link_libraries to add dependencies to a "target"
# (e.g., a library or executable)
# This will automatically add all required library files
# that need to be linked
# and paths to th locations of header files
target_link_libraries(reconstruct Eigen3::Eigen ${OpenCV_LIBS} Ceres::ceres)
target_link_libraries(visualizer imgui glad)
target_include_directories(reconstruct
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>)