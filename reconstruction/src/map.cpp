#include "reconstruction/map.hpp"

namespace sfm
{
MapPoint::MapPoint(cv::Mat descriptor, cv::Mat3d position, size_t keyframe_id)
{
  desc = descriptor;
  pos = position;
  keyframe_ids.push_back(keyframe_id);
}
}
