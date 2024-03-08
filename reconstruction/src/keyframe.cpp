#include "reconstruction/pinhole.hpp"
#include "reconstruction/features.hpp"
#include "reconstruction/mappoint.hpp"
#include "reconstruction/keyframe.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <ranges>
#include <iostream>

namespace sfm
{
KeyFrame::KeyFrame(
  PinholeModel K, Eigen::Matrix4d T_wk, std::vector<cv::KeyPoint> keypoints,
  cv::Mat descriptions, cv::Mat img, cv::Mat depth, PinholeModel model)
: K{K}, T_wk{T_wk}, T_kw{T_wk.inverse()}, keypoints{keypoints}, descriptors{descriptions}, img{img},
  depth_img{depth}
{
  // convert descriptor mat to vector
  for (int i = 0; i < descriptions.rows; i++) {
    descriptors_vec.push_back(descriptions.row(i));
  }

  deprojected_points = deproject_keypoints(keypoints, depth, model);
}

cv::Mat KeyFrame::get_depth_image() const
{
  return depth_img;
}

cv::Mat KeyFrame::get_img() const
{
  return img;
}

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

std::vector<cv::Mat> KeyFrame::get_descriptors() const
{
  return descriptors_vec;
}

cv::Mat KeyFrame::get_descriptors_mat() const
{
  return descriptors;
}

DBoW2::BowVector KeyFrame::get_bow_vector() const
{
  return bow_vector;
}

void KeyFrame::set_bow_vector(const DBoW2::BowVector & bow_vector)
{
  this->bow_vector = bow_vector;
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

bool KeyFrame::link_map_point(size_t kp_idx, std::shared_ptr<MapPoint> map_point)
{
  if (kp_to_mp_index.find(kp_idx) != kp_to_mp_index.end()) {
    std::cout << "DUPLICATES KP IDX\n";
  }

  if (mp_to_kp_index.find(map_point) != mp_to_kp_index.end()) {
    std::cout << "DUPLICATES MP\n";
    return false;
  }

  kp_to_mp_index[kp_idx] = map_point;
  mp_to_kp_index[map_point] = kp_idx;
  return true;
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

      //2. get color
      const auto color = extract_color(img, pt);

      //3. make map point
      std::shared_ptr<KeyFrame> wp = shared_from_this();
      auto map_point = std::make_shared<MapPoint>(
        descriptors.row(i),
        (T_wk * deprojected_points[i].homogeneous()).head<3>(),
        wp, color);

      link_map_point(i, map_point);

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

std::optional<cv::Mat> KeyFrame::get_descriptor(std::shared_ptr<MapPoint> map_point) const
{
  if (mp_to_kp_index.find(map_point) == mp_to_kp_index.end()) {
    return {};
  }
  return descriptors.row(mp_to_kp_index.at(map_point));
}

std::vector<size_t> KeyFrame::get_features_within_radius(
  double x, double y, double r,
  bool allow_matched)
{
  const auto r2 = r * r;
  std::vector<size_t> indicies;
  for (size_t i = 0; i < keypoints.size(); i++) {
    if (allow_matched || kp_to_mp_index.find(i) == kp_to_mp_index.end()) {
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
  return {static_cast<int>(kp.pt.x), static_cast<int>(kp.pt.y)};
}

Eigen::Vector3d KeyFrame::get_observed_location_3d(const std::shared_ptr<MapPoint> map_point) const
{
  const auto idx = mp_to_kp_index.at(map_point);
  return (T_wk * deprojected_points[idx].homogeneous()).head<3>();
}

void KeyFrame::remove_map_point(const std::shared_ptr<MapPoint> map_point)
{
  const auto idx = mp_to_kp_index.at(map_point);
  kp_to_mp_index.erase(idx);
  mp_to_kp_index.erase(map_point);
}

cv::Point2d project_map_point(const KeyFrame & key_frame, const MapPoint & map_point)
{
  return project_pixel_to_point(
    key_frame.camera_calibration(),
    key_frame.world_to_camera(), map_point.position());
}

double keyframe_reprojection_error(
  const KeyFrame & key_frame, const std::shared_ptr<MapPoint> map_point,
  const Eigen::Vector3d & world_point)
{
  const auto point = project_pixel_to_point(
    key_frame.camera_calibration(), key_frame.world_to_camera(), world_point);
  const auto [x, y] = key_frame.get_observed_location(map_point);
  return std::sqrt((x - point.x) * (x - point.x) + (y - point.y) * (y - point.y));
}

bool KeyFrame::point_in_frame(const MapPoint & point) const
{
  const auto p = project_pixel_to_point(
    K, T_kw, point.position());

  // is within image bounds
  if (p.x < 0 || p.x > img.cols) {
    return false;
  }

  if (p.y < 0 || p.y > img.rows) {
    return false;
  }

  // is in front of the camera
  const auto camera_point = T_kw * point.position().homogeneous();
  if (camera_point.z() < 0) {
    return false;
  }

  // get the mean viewing ray of the point
  Eigen::Vector3d mean_ray = Eigen::Vector3d::Zero();
  for (const auto & kf: point.get_keyframes()) {
    const auto shared_kf = kf.lock();
    const Eigen::Vector3d vec =
      (shared_kf->world_to_camera() * point.position().homogeneous()).head<3>().normalized();
    mean_ray += vec;
  }
  mean_ray /= point.get_keyframes().size();
  mean_ray.normalize();

  Eigen::Vector3d ray = (T_kw * point.position().homogeneous()).head<3>().normalized();

  if (mean_ray.dot(ray) < std::cos(1.0472)) {
    return false;
  }

  return true;
}

std::optional<cv::Point2i> KeyFrame::point_in_frame(const Eigen::Vector3d & point) const
{
  auto p = project_pixel_to_point(K, T_kw, point);

  // is within image bounds
  if (p.x < 0 || p.x >= img.cols || p.y < 0 || p.y >= img.rows) {
    return {};
  }

  // is in front of the camera
  const auto camera_point = T_kw * point.homogeneous();
  if (camera_point.z() < 0) {
    return {};
  }

  return cv::Point2i{
    static_cast<int>(std::floor(p.x)),
    static_cast<int>(std::floor(p.y))
  };
}
}
