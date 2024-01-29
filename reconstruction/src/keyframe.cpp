#include "reconstruction/keyframe.hpp"
#include "reconstruction/pinhole.hpp"
#include <algorithm>
#include <opencv2/features2d.hpp>

namespace sfm
{
KeyFrame::KeyFrame(
  PinholeModel K, Eigen::Matrix4d T_kw, std::vector<cv::KeyPoint> keypoints,
  cv::Mat descriptions, cv::Mat img)
: K{K}, T_kw{T_kw}, keypoints{keypoints}, descriptors{descriptions}, img{img}
{}

std::vector<cv::DMatch> KeyFrame::match(
  const cv::Mat & query_descriptiors,
  const std::shared_ptr<cv::DescriptorMatcher> matcher) const
{
  std::vector<std::vector<cv::DMatch>> kmatches;
  matcher->knnMatch(query_descriptiors, descriptors, kmatches, 2);

  //ratio test https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
  std::vector<cv::DMatch> good_matches;
  for (const auto & matches: kmatches) {
    const auto & m1 = matches[0];
    const auto & m2 = matches[1];

    if (m1.distance < 0.75 * m2.distance) {
      good_matches.push_back(m1);
    }
  }

  return good_matches;
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
