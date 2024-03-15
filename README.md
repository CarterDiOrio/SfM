# A package to reconstruct the environment from a sequence of RGBD images.

- Author: Carter DiOrio
- Portfolio Post: https://www.cdiorio.dev/projects/sfm/
- Initially Published: 3/15

## Requirements
- C++23
- Eigen
- Sophus
- Ceres
- OpenCV
- Boost

## Reconstruct

Reconstruct is the main executable built by this project. 
- It can be run with `reconstruct --data <path to data directory> --vocab <path to DBoW2 vocab file>`
- The data directory contains a series of numbered `color_N` and `depth_N` images from 0 to N.
- The vocabulary file needs to be compatible with DBoW2 see https://github.com/dorian3d/DBoW2 for how to generate one. 
- Outputs a file with a dense `X Y Z R G B` point cloud file
