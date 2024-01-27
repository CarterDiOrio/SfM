#include "reconstruction/map.hpp"

namespace sfm
{
MapPoint::MapPoint(cv::Mat descriptor, cv::Mat3d position, size_t keyframe_id)
{
  desc = descriptor;
  pos = position;
  keyframe_ids.push_back(keyframe_id);
}

Map::Map()
{}


bool Map::is_empty()
{
  return keyframes.size() == 0;
}

size_t Map::add_keyframe(const KeyFrame & frame)
{
  keyframes.push_back(frame);
  return keyframes.size() - 1;
}

}
