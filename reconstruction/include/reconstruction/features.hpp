#ifndef INC_GUARD_FEATURES_HPP
#define INC_GUARD_FEATURES_HPP

#include "keyframe.hpp"
#include <opencv2/features2d.hpp>

namespace sfm
{

/// @brief Detects features and computes their descriptions within the img
/// @param img The image to detect ORB features in
/// @return a pair of two vectors {keypoints and descriptions}
std::pair<std::vector<cv::KeyPoint>, cv::Mat> detect_features(
  const cv::Mat & img, std::shared_ptr<cv::ORB> feature_detector
);

/// @brief Deprojects each keypoint to a 3D point in the camera frame
/// @param keypoints the keypoints to deproject
/// @param depth the depth image
/// @param model the pinhole camera model
/// @return a vector of 3x1 vectors for each 3D point
std::vector<Eigen::Vector3d> deproject_keypoints(
  const std::vector<cv::KeyPoint> & keypoints, const cv::Mat & depth, const PinholeModel & model);

/// @brief Extracts the color value from a mat
/// @param frame the color frame
/// @param keypoints the keypoints to get the colors at
/// @return a vector of eigen 3x1 vectors representing the colors
std::vector<Eigen::Vector3i> extract_colors(
  const cv::Mat & frame,
  const std::vector<cv::KeyPoint> & keypoints
);

/// @brief Extracts the color for a given point
/// @param frame the color frame
/// @param point the point to extract the color at
/// @return an eigen 3x1 vector representing the color
Eigen::Vector3i extract_color(const cv::Mat & frame, const cv::Point2d & point);
}


#endif
