#ifndef INC_GUARD_PINHOLE_HPP
#define INC_GUARD_PINHOLE_HPP

#include <opencv2/core.hpp>

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>


namespace sfm::model
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

/// @brief Deprojects a pixel to a 3D point
/// @param model The pinhole camera model
/// @param point the pixel to deproject
/// @param depth the depth of the point in meters
/// @return a 3x1 column vector of the 3D point
Eigen::Vector3d deproject_pixel_to_point(
  const PinholeModel & model, cv::Point2d point,
  double depth);

/// @brief Projects a 3D point to the image plane
/// @param model the camera model
/// @param world_to_camera_tf the transform from the world to the camera frame
/// @param point the point to project
/// @return a point in the image plane
cv::Point2d project_point_to_pixel(
  const PinholeModel & model,
  const Eigen::Matrix4d & world_to_camera_tf,
  const Eigen::Vector3d & point);

/// @brief Convers the PinholeModel to an opencv matrix
/// @param model the camera model
/// @return an opencv mat that is the 3x3 camera calibration matrix
cv::Mat model_to_mat(const PinholeModel & model);

/// @brief Converts the PinholeModel to an Eigen matrix
/// @param model the camera model
/// @return a 3x3 Eigen matrix of the camera calibration
Eigen::Matrix3d model_to_eigen(const PinholeModel & model);

/// @brief Performs Perspective-n-Point between the image points and the world points
/// @param image_points the 2D points in the image
/// @param world_points the corresponding 3D world peoints
/// @param model the camera model
/// @return the 4x4 transformation matrix of the camera to the world
std::pair<Eigen::Matrix4d, std::vector<int>> pnp(
  const std::vector<cv::Point2d> & image_points,
  const std::vector<Eigen::Vector3d> & world_points,
  const PinholeModel & model);
}

#endif
