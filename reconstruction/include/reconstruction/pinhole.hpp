#ifndef INC_GUARD_PINHOLE_HPP
#define INC_GUARD_PINHOLE_HPP

#include <opencv2/core.hpp>

#include <Eigen/Dense>

namespace sfm
{
/// @brief Camera calibration values for pinhole camera.
struct PinholeModel
{
  /// @brief The x focal length in pixels
  double fx;

  /// @brief The y focal length in pixels
  double fy;

  /// @brief The actual camera center x coordinate
  double cx;

  /// @brief The actual camera center y coorindate
  double cy;
};


/// @brief Deprojects a pixel to a 3D point
/// @param model The pinhole camera model
/// @param px the x coordinate in the image plane
/// @param py the y coordinate in the image plane
/// @param depth the depth of the point in meters
/// @return a 3x1 column vector of the 3D point
Eigen::Vector3d deproject_pixel_to_point(const PinholeModel & model, int px, int py, double depth);

}

#endif
