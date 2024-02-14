#include "reconstruction/pinhole.hpp"
#include "reconstruction/features.hpp"
#include "reconstruction/mappoint.hpp"
#include "reconstruction/keyframe.hpp"

#include <cstdint>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <ranges>
#include <iostream>

namespace sfm
{
KeyFrame::KeyFrame(
  PinholeModel K, Eigen::Matrix4d T_wk, std::vector<cv::KeyPoint> keypoints,
  cv::Mat descriptions, cv::Mat img, cv::Mat depth)
: K{K}, T_wk{T_wk}, T_kw{T_wk.inverse()}, keypoints{keypoints}, descriptors{descriptions}, img{img},
  depth_img{depth}
{}

PinholeModel KeyFrame::camera_calibration() const
{
  return K;
}

Eigen::Matrix4d KeyFrame::transform() const
{
  return T_wk;
}

Eigen::Matrix4d KeyFrame::world_to_camera() const
{
  return T_kw;
}

void KeyFrame::set_world_to_camera(const Eigen::Matrix4d tf)
{
  T_wk = tf.inverse();
  T_kw = tf;
}

size_t KeyFrame::num_keypoints() const
{
  return keypoints.size();
}

std::pair<cv::KeyPoint, cv::Mat> KeyFrame::get_point(int i) const
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
  std::vector<cv::DMatch> matches;
  matcher->match(query_descriptiors, descriptors, matches);
  return matches;
}

void KeyFrame::link_map_point(size_t kp_idx, std::shared_ptr<MapPoint> map_point)
{
  if (kp_to_mp_index.find(kp_idx) != kp_to_mp_index.end()) {
    std::cout << "DUPLICATES KP IDX\n";
  }

  if (mp_to_kp_index.find(map_point) != mp_to_kp_index.end()) {
    std::cout << "DUPLICATES MP\n";
  }

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
      const double depth = depth_img.at<uint16_t>(pt);
      const auto point3d = deproject_pixel_to_point(K, pt.x, pt.y, depth);
      const auto world_point = (T_wk * point3d.homogeneous()).head<3>();

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

std::vector<std::shared_ptr<MapPoint>> KeyFrame::get_map_points() const
{
  auto mp_view = mp_to_kp_index | std::views::keys;
  return {
    mp_view.begin(),
    mp_view.end()
  };
}

std::vector<size_t> KeyFrame::get_features_within_radius(double x, double y, double r)
{
  const auto r2 = r * r;
  std::vector<size_t> indicies;
  for (size_t i = 0; i < keypoints.size(); i++) {
    if (kp_to_mp_index.find(i) == kp_to_mp_index.end()) {
      const auto & kp = keypoints[i];
      const auto dx = kp.pt.x - x;
      const auto dy = kp.pt.y - y;
      if (dx * dx + dy * dy < r2) {
        indicies.push_back(i);
      }
    }
  }
  return indicies;
}

std::pair<double,
  double> KeyFrame::get_observed_location(const std::shared_ptr<MapPoint> map_point) const
{
  const auto idx = mp_to_kp_index.at(map_point);
  const auto & kp = keypoints[idx];
  return {kp.pt.x, kp.pt.y};
}

cv::Point2d project_map_point(const KeyFrame & key_frame, const MapPoint & map_point)
{
  return project_pixel_to_point(
    key_frame.camera_calibration(),
    key_frame.world_to_camera(), map_point.position());
}

}
