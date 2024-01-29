#include "reconstruction/map.hpp"

#include <algorithm>
#include <pcl/filters/frustum_culling.h>
#include <memory>
#include <iostream>

namespace sfm
{
MapPoint::MapPoint(
  cv::Mat descriptor, Eigen::Vector3d position, size_t keyframe_id,
  Eigen::Vector3i color)
: color{color}
{
  desc = descriptor;
  pos = position;
  keyframe_ids.push_back(keyframe_id);
}

void MapPoint::add_keyframe(size_t k_id)
{
  keyframe_ids.push_back(k_id);
}

Map::Map() {}

bool Map::is_empty()
{
  return keyframes.size() == 0;
}

size_t Map::add_keyframe(const KeyFrame & frame)
{
  keyframes.push_back(frame);
  return keyframes.size() - 1;
}

void Map::create_mappoints(
  size_t k_id,
  const std::vector<Eigen::Vector3d> points,
  const std::vector<Eigen::Vector3i> colors,
  const cv::Mat & descriptions,
  const std::vector<size_t> orig_idx = std::vector<size_t>{})
{
  auto & keyframe = keyframes.at(k_id);

  // transform the points from the camera frame to the world frame
  std::vector<Eigen::Vector3d> world_points(points.size());
  std::transform(
    points.begin(), points.end(), world_points.begin(),
    [&keyframe](const Eigen::Vector3d & point) {
      Eigen::Vector3d world = (keyframe.transform() * point.homogeneous()).head<3>();
      return world;
    }
  );

  // add map points to list and point cloud
  for (size_t i = 0; i < points.size(); i++) {
    MapPoint mp{descriptions.row(i), world_points.at(i), k_id, colors[i]};

    size_t idx = i;
    if (orig_idx.size() > 0) {
      idx = orig_idx[i];
    }

    keyframe.link_map_point(idx, mappoints.size());
    mappoints.push_back(mp);
  }
}

std::ostream & operator<<(std::ostream & os, const Map & map)
{
  for (const auto & map_point: map.mappoints) {
    const auto pos = map_point.position();
    const auto color = map_point.get_color();
    os << pos(0) << ", " << pos(1) << ", " << pos(2) << ", " << color(2) << ", " << color(1) <<
      ", " << color(0) << "\n";
  }
  return os;
}

}
