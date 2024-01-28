#include "reconstruction/map.hpp"

#include <algorithm>
#include <pcl/filters/frustum_culling.h>
#include <memory>

namespace sfm
{
MapPoint::MapPoint(cv::Mat descriptor, Eigen::Vector3d position, size_t keyframe_id)
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
  size_t k_id, const std::vector<Eigen::Vector3d> points,
  const std::vector<cv::Mat> descriptions)
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
    MapPoint mp{descriptions.at(i), world_points.at(i), k_id};
    keyframe.link_map_point(i, mappoints.size());
    mappoints.push_back(mp);
  }
}


}
