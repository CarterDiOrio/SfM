#include "reconstruction/keyframe.hpp"
#include "reconstruction/pinhole.hpp"
#include <algorithm>
#include <opencv2/features2d.hpp>

namespace sfm
{
KeyFrame::KeyFrame(
  PinholeModel K, Eigen::Matrix4d T_kw, std::vector<cv::KeyPoint> keypoints,
  std::vector<cv::Mat> descriptions)
: K{K}, T_kw{T_kw}, keypoints{keypoints}, descriptors{descriptions}
{}

std::vector<cv::DMatch> KeyFrame::match(
  const std::vector<cv::Mat> & query_descriptiors,
  const cv::DescriptorMatcher & matcher) const
{
  std::vector<cv::DMatch> matches;
  matcher.match(query_descriptiors, descriptors, matches);
  return matches;
}

void KeyFrame::link_map_point(size_t kp_idx, size_t map_point_id)
{
  kp_to_mp_index[kp_idx] = map_point_id;
  mp_to_kp_index[map_point_id] = kp_idx;
}

std::optional<size_t> KeyFrame::corresponding_map_point(size_t keypoint_idx) const
{
  if (kp_to_mp_index.find(keypoint_idx) == kp_to_mp_index.end()) {
    return {};
  }
  return kp_to_mp_index.at(keypoint_idx);
}
}
