#include "reconstruction/pinhole.hpp"
#include "reconstruction/features.hpp"
#include "reconstruction/mappoint.hpp"
#include "reconstruction/keyframe.hpp"

#include <algorithm>
#include <opencv2/features2d.hpp>

namespace sfm
{
KeyFrame::KeyFrame(
  PinholeModel K, Eigen::Matrix4d T_kw, std::vector<cv::KeyPoint> keypoints,
  cv::Mat descriptions, cv::Mat img, cv::Mat depth)
: K{K}, T_kw{T_kw}, keypoints{keypoints}, descriptors{descriptions}, img{img}, depth_img{depth}
{}

PinholeModel KeyFrame::camera_calibration() const
{
  return K;
}

Eigen::Matrix4d KeyFrame::transform() const
{
  return T_kw;
}

size_t KeyFrame::num_keypoints() const
{
  return keypoints.size();
}

std::pair<cv::KeyPoint, cv::Mat> KeyFrame::operator[](int i) const
{
  return {
    keypoints[i],
    descriptors.row(i)
  };
}

const std::vector<cv::KeyPoint> & KeyFrame::get_keypoints() const
{
  return keypoints;
}

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

void KeyFrame::link_map_point(size_t kp_idx, std::shared_ptr<MapPoint> map_point)
{
  kp_to_mp_index[kp_idx] = map_point;
  mp_to_kp_index[map_point] = kp_idx;
}

std::optional<std::shared_ptr<MapPoint>> KeyFrame::corresponding_map_point(size_t keypoint_idx)
const
{
  if (kp_to_mp_index.find(keypoint_idx) == kp_to_mp_index.end()) {
    return {};
  }
  return kp_to_mp_index.at(keypoint_idx);
}

std::vector<std::pair<size_t, std::shared_ptr<MapPoint>>> KeyFrame::create_map_points()
{
  std::vector<std::pair<size_t, std::shared_ptr<MapPoint>>> map_points;

  for (size_t i = 0; i < keypoints.size(); i++) {
    if (kp_to_mp_index.find(i) == kp_to_mp_index.end()) {
      const auto & pt = keypoints[i].pt;

      //1. deproject the point
      const uint16_t depth = depth_img.at<uint16_t>(pt);
      const auto point3d = deproject_pixel_to_point(K, pt.x, pt.y, depth);
      const auto world_point = (T_kw * point3d.homogeneous()).head<3>();

      //2. get color
      const auto color = extract_color(img, pt);

      //3. make map point
      std::shared_ptr<KeyFrame> wp = shared_from_this();
      auto map_point = std::make_shared<MapPoint>(
        descriptors.row(i), world_point,
        wp, color);

      map_points.push_back({i, map_point});
    }
  }

  return map_points;
}
}
