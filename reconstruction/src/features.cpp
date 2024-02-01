#include "reconstruction/features.hpp"
#include <opencv2/core/hal/interface.h>
#include <opencv2/features2d.hpp>
#include <ranges>

namespace sfm
{

Features detect_features(
  const cv::Mat & img, std::shared_ptr<cv::ORB> feature_detector
)
{
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  feature_detector->detect(img, keypoints);
  feature_detector->compute(img, keypoints, descriptors);
  return {
    keypoints,
    descriptors
  };
}

std::vector<Eigen::Vector3d> deproject_keypoints(
  const std::vector<cv::KeyPoint> & keypoints, const cv::Mat & depth, const PinholeModel & model)
{
  std::vector<Eigen::Vector3d> points(keypoints.size());
  std::transform(
    keypoints.begin(), keypoints.end(), points.begin(),
    [&depth, &model](const cv::KeyPoint & kp) {
      auto pixel_depth = depth.at<uint16_t>(kp.pt);
      return deproject_pixel_to_point(model, kp.pt.x, kp.pt.y, pixel_depth);
    }
  );
  return points;
}

Features filter_features(
  const Features & features,
  const cv::Mat & depth_img,
  double max_depth)
{
  const auto & [keypoints, descriptors] = features;
  std::vector<cv::KeyPoint> filtered_keypoints;
  cv::Mat filtered_descriptors;

  for (const auto [idx, kp]: std::views::enumerate(keypoints)) {
    auto depth = depth_img.at<uint16_t>(kp.pt);
    if (depth < max_depth) {
      filtered_keypoints.push_back(keypoints[idx]);
      filtered_descriptors.push_back(descriptors.row(idx));
    }
  }

  return {
    .keypoints = filtered_keypoints,
    .descriptors = filtered_descriptors,
  };
}

std::vector<Eigen::Vector3i> extract_colors(
  const cv::Mat & frame,
  const std::vector<cv::KeyPoint> & keypoints
)
{
  std::vector<Eigen::Vector3i> colors(keypoints.size());
  std::transform(
    keypoints.begin(), keypoints.end(), colors.begin(),
    [&frame](const cv::KeyPoint & keypoint) {
      const auto vec = frame.at<cv::Vec3b>(keypoint.pt);
      return Eigen::Vector3i{(int)vec[0], (int)vec[1], (int)vec[2]};
    }
  );
  return colors;
}

Eigen::Vector3i extract_color(const cv::Mat & frame, const cv::Point2d & point)
{
  const auto color_vec = frame.at<cv::Vec3b>(point);
  return {(int)color_vec[0], (int)color_vec[1], (int)color_vec[2]};
}

}
