#include "reconstruction/mappoint.hpp"
#include "reconstruction/keyframe.hpp"

#include <memory>
#include <ranges>

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
