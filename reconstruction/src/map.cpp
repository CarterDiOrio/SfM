#include "reconstruction/map.hpp"

#include <algorithm>
#include <pcl/filters/frustum_culling.h>
#include <memory>
#include <iostream>

#include "reconstruction/mappoint.hpp"
#include "reconstruction/keyframe.hpp"

namespace sfm
{

Map::Map() {}

bool Map::is_empty()
{
  return keyframes.size() == 0;
}

std::weak_ptr<KeyFrame> Map::create_keyframe(
  PinholeModel K,
  Eigen::Matrix4d T_kw,
  std::vector<cv::KeyPoint> keypoints,
  cv::Mat descriptions,
  cv::Mat img,
  cv::Mat depth)
{
  const auto new_keyframe =
    std::make_shared<KeyFrame>(K, T_kw, keypoints, descriptions, img, depth);
  keyframes.push_back(new_keyframe);
  return new_keyframe;
}

size_t Map::add_keyframe(std::shared_ptr<KeyFrame> keyframe)
{
  keyframes.push_back(keyframe);
  return keyframes.size() - 1;
}

void Map::add_map_point(std::shared_ptr<MapPoint> map_point)
{
  mappoints.push_back(map_point);
}

std::ostream & operator<<(std::ostream & os, const Map & map)
{
  for (const auto & map_point: map.mappoints) {
    const auto pos = map_point->position();
    const auto color = map_point->get_color();
    os << pos(0) << ", " << pos(1) << ", " << pos(2) << ", " << color(2) << ", " << color(1) <<
      ", " << color(0) << "\n";
  }
  return os;
}

}
