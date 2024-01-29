#include "reconstruction/pinhole.hpp"
#include <iostream>


namespace sfm
{
Eigen::Vector3d deproject_pixel_to_point(const PinholeModel & model, int px, int py, double depth)
{
  /// convert pixel to canonical camera plane, can be projected to depth by
  /// multiplication from here
  auto x = (px - model.cx) / model.fx;
  auto y = (py - model.cy) / model.fy;

  Eigen::Vector3d point {
    x * depth,
    y * depth,
    depth
  };

  // that point is where the z axis is forward
  // Eigen::Matrix3d t;
  // t <<
  //   0.0, 0.0, 1.0,
  //   0.0, 1.0, 0.0,
  //   1.0, 0.0, 0.0;

  // Eigen::Vector3d deprojected = t * point;
  return point;
}

cv::Mat model_to_mat(const PinholeModel & model)
{
  cv::Mat mat =
    (cv::Mat_<double>(3, 3) << model.fx, 0.0, model.cx, 0.0, model.fy, model.cy, 0.0, 0.0,
    1.0);
  return mat;
}
}
