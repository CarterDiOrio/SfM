#include "reconstruction/map.hpp"

#include <algorithm>
#include <opencv2/core.hpp>
#include <pcl/filters/frustum_culling.h>
#include <memory>
#include <iostream>
#include <deque>
#include <ranges>

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

void Map::link_keyframe_to_map_point(
  std::shared_ptr<KeyFrame> key_frame,
  size_t key_point_idx,
  std::shared_ptr<MapPoint> map_point)
{
  key_frame->link_map_point(key_point_idx, map_point);

  // link covisibility graph
  for (auto kf: *map_point) {
    auto shared_kf = kf.lock();
    if (key_frame != shared_kf) {
      covisibility_insert(key_frame, shared_kf);
    }
  }
}

void Map::covisibility_insert(
  std::shared_ptr<KeyFrame> key_frame_1,
  std::shared_ptr<KeyFrame> key_frame_2)
{
  const auto insert = [&covisibility = covisibility](auto kf1, auto kf2) {
      if (covisibility.find(kf1) == covisibility.end()) {
        covisibility[kf1] = {kf2};
      } else {
        auto & vec = covisibility[kf1];
        if (std::find(vec.begin(), vec.end(), kf2) == vec.end()) {
          vec.push_back(kf2);
        }
      }
    };
  insert(key_frame_1, key_frame_2);
  insert(key_frame_2, key_frame_1);
}

std::vector<key_frame_set_t> Map::get_local_map(
  std::shared_ptr<KeyFrame> key_frame,
  size_t distance
)
{
  std::vector<key_frame_set_t> sets;
  std::vector<std::shared_ptr<KeyFrame>> visited;

  auto vec = covisibility[key_frame];
  std::deque<std::shared_ptr<KeyFrame>> key_frame_queue{vec.begin(), vec.end()};
  sets.push_back({vec.begin(), vec.end()});

  // filter key frames for not in queue
  const auto visited_filter = [&visited](auto kf) {
      return std::find(
        visited.begin(),
        visited.end(),
        kf
      ) != visited.end();
    };

  for (size_t d = 1; d < distance; d++) {

    //get length of all nodes in layer
    size_t layer_size = key_frame_queue.size();

    // process all nodes at current layer
    for (size_t i = 0; i < layer_size; i++) {
      const auto current = key_frame_queue.front();

      // get the nodes that haven't been visited
      for (auto kf: covisibility[current] | std::views::filter(visited_filter)) {
        visited.push_back(kf);
        key_frame_queue.push_back(kf);
      }
      key_frame_queue.pop_front();
    }

    sets.push_back({key_frame_queue.begin(), key_frame_queue.end()});
  }

  return sets;
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
