#ifndef INC_GUARD_PINHOLE_HPP
#define INC_GUARD_PINHOLE_HPP

#include <opencv2/core.hpp>

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

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

/// @brief Projects a pixel to the image plane
/// @param model the camera model
/// @param transform the transform from the world to the camera frame
/// @param world_point the point to project
/// @return a point in the image plane
cv::Point2d project_pixel_to_point(
  const PinholeModel & model,
  const Eigen::Matrix4d transform,
  const Eigen::Vector3d & world_point);

/// @brief Convers the PinholeModel to an opencv matrix
/// @param model the camera model
/// @return an opencv mat that is the 3x3 camera calibration matrix
cv::Mat model_to_mat(const PinholeModel & model);


}

#endif
