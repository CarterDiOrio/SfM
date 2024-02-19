#include "reconstruction/mappoint.hpp"
#include "reconstruction/keyframe.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <ranges>
#include <iostream>

namespace sfm
{
MapPoint::MapPoint(
  cv::Mat descriptor,
  Eigen::Vector3d position,
  std::weak_ptr<KeyFrame> keyframe,
  Eigen::Vector3i color)
{
  desc = descriptor;
  pos = position;
  keyframes.push_back(keyframe);
  this->color = color;
}

void MapPoint::add_keyframe(std::weak_ptr<KeyFrame> keyframe)
{
  keyframes.push_back(keyframe);

  // get all of the descriptors and 3d locations
  std::vector<cv::Mat> descriptors;
  std::vector<Eigen::Vector3d> positions;
  for (const auto & kf: keyframes) {
    auto shared_kf = kf.lock();
    auto descriptor = shared_kf->get_descriptor(shared_from_this());

    descriptors.push_back(descriptor.value());
    positions.push_back(shared_kf->get_observed_location_3d(shared_from_this()));
  }

  // find the best descriptor
  std::vector<double> scores;
  for (const auto & desc: descriptors) {
    scores.push_back(0.0);
    for (const auto & other_desc: descriptors) {
      scores[scores.size() - 1] += cv::norm(desc, other_desc, cv::NORM_HAMMING);
    }
  }
  size_t idx = std::distance(scores.begin(), std::min_element(scores.begin(), scores.end()));

  desc = descriptors[idx];

  update_position();
}

void MapPoint::update_position()
{
  std::vector<Eigen::Vector3d> positions;
  for (const auto & kf: keyframes) {
    auto shared_kf = kf.lock();
    positions.push_back(shared_kf->get_observed_location_3d(shared_from_this()));
  }

  // find the 3D location with the lowest reprojection error
  std::vector<double> reprojection_errors;
  for (const auto & observed_pos: positions) {
    double err = 0.0;
    for (const auto & kf: keyframes) {
      auto shared_kf = kf.lock();
      err += keyframe_reprojection_error(
        *shared_kf,
        shared_from_this(), observed_pos);
    }
    reprojection_errors.push_back(err);
  }

  auto idx = std::distance(
    reprojection_errors.begin(),
    std::min_element(reprojection_errors.begin(), reprojection_errors.end()));

  pos = positions[idx];
}

void MapPoint::remove_keyframe(std::weak_ptr<KeyFrame> keyframe)
{
  keyframes.erase(
    std::remove_if(
      keyframes.begin(),
      keyframes.end(),
      [keyframe](const auto & kf) {
        return kf.lock() == keyframe.lock();
      }),
    keyframes.end());
}

std::vector<std::weak_ptr<KeyFrame>> MapPoint::get_keyframes()
{
  return keyframes;
}

Eigen::Vector3d MapPoint::position() const
{
  return pos;
}

cv::Mat MapPoint::description() const
{
  return desc;
}

Eigen::Vector3i MapPoint::get_color() const
{
  return color;
}

MapPoint::key_frames_t::iterator MapPoint::begin()
{
  return keyframes.begin();
}

MapPoint::key_frames_t::iterator MapPoint::end()
{
  return keyframes.end();
}
}
