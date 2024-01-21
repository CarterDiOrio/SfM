# Lines that begin with a # are comments
# set the minimum required version of cmake, usually the first line
cmake_minimum_required(VERSION 3.22)

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


# Create an executable from the following source code files
# The Name of the executable creates a cmake "target"
add_executable(essential src/essential.cpp)
add_executable(reconstruct src/reconstruct.cpp)

# Enable c++17 support.
# Public causes the features to propagate to anything
target_compile_features(essential PUBLIC cxx_std_17)

# Use target_link_libraries to add dependencies to a "target"
# (e.g., a library or executable)
# This will automatically add all required library files
# that need to be linked
# and paths to th locations of header files
target_link_libraries(essential Eigen3::Eigen ${OpenCV_LIBS})
target_link_libraries(reconstruct Eigen3::Eigen ${OpenCV_LIBS})

target_include_directories(essential
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>)