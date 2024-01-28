#include "reconstruction/features.hpp"
#include <opencv2/features2d.hpp>

namespace sfm
{

std::pair<std::vector<cv::KeyPoint>, std::vector<cv::Mat>> detect_features(
  const cv::Mat & img, std::shared_ptr<cv::ORB> feature_detector
)
{
  std::vector<cv::KeyPoint> keypoints;
  std::vector<cv::Mat> descriptors;
  feature_detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
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
      auto pixel_depth = depth.at<uint16_t>(
        (int)std::round(kp.pt.x),
        (int)std::round(kp.pt.y));
      return deproject_pixel_to_point(model, kp.pt.x, kp.pt.y, pixel_depth);
    }
  );
  return points;
}
}
