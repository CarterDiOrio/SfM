#include "reconstruction/pinhole.hpp"
#include <iostream>
#include <opencv2/core/types.hpp>


namespace sfm::model
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

  return point;
}

cv::Point2d project_point_to_pixel(
  const PinholeModel & model,
  const Eigen::Matrix4d & world_to_camera_tf,
  const Eigen::Vector3d & point)
{
  const Eigen::Vector4d point_camera = world_to_camera_tf * point.homogeneous();
  return {
    (point_camera(0) / point_camera(2)) * model.fx + model.cx,
    (point_camera(1) / point_camera(2)) * model.fy + model.cy
  };
}

Eigen::Vector3d deproject_pixel_to_point(
  const PinholeModel & model, cv::Point2d point,
  double depth)
{
  return deproject_pixel_to_point(model, point.x, point.y, depth);
}

cv::Mat model_to_mat(const PinholeModel & model)
{
  cv::Mat mat =
    (cv::Mat_<double>(3, 3) << model.fx, 0.0, model.cx, 0.0, model.fy, model.cy, 0.0, 0.0,
    1.0);
  return mat;
}

Eigen::Matrix3d model_to_eigen(const PinholeModel & model)
{
  Eigen::Matrix3d mat;
  mat << model.fx, 0.0, model.cx, 0.0, model.fy, model.cy, 0.0, 0.0, 1.0;
  return mat;
}

std::pair<Eigen::Matrix4d, std::vector<int>> pnp(
  const std::vector<cv::Point2d> & image_points,
  const std::vector<Eigen::Vector3d> & world_points,
  const PinholeModel & model)
{
  std::vector<cv::Point3d> world_points_cv(world_points.size());
  std::transform(
    world_points.begin(), world_points.end(), world_points_cv.begin(),
    [](const Eigen::Vector3d & vec) {
      return cv::Point3d{vec(0), vec(1), vec(2)};
    }
  );

  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);

  std::vector<int> inliers;
  cv::Mat k = model_to_mat(model);
  cv::solvePnPRansac(
    world_points_cv, image_points, k,
    cv::noArray(), rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_SQPNP);

  // convert screw to transformation matrix
  cv::Mat rotation_mat;
  cv::Rodrigues(rvec, rotation_mat);

  Eigen::Matrix3d rotation_eig;
  cv::cv2eigen(rotation_mat, rotation_eig);
  Eigen::Vector3d translation_eig{tvec.at<double>(0, 0), tvec.at<double>(1, 0),
    tvec.at<double>(2, 0)};

  // the transformation is given from world to camera and we want camera to world
  Eigen::Matrix4d transformation;
  transformation.setIdentity();
  transformation.block<3, 3>(0, 0) = rotation_eig.transpose();
  transformation.block<3, 1>(0, 3) = -1 * rotation_eig.transpose() * translation_eig;
  return {transformation, inliers};
}
}
