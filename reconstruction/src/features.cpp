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
}
