#include "reconstruction/map.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/AngleAxis.h>
#include <algorithm>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/types.h>
#include <opencv2/core.hpp>
#include <pcl/filters/frustum_culling.h>
#include <memory>
#include <iostream>
#include <deque>
#include <ranges>
#include <unordered_map>
#include <unordered_set>

#include "Eigen/Geometry"
#include "reconstruction/mappoint.hpp"
#include "reconstruction/keyframe.hpp"
#include "reconstruction/bundle_adjust.hpp"
#include "reconstruction/pinhole.hpp"

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
  covisibility[new_keyframe] = {};
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
  map_point->add_keyframe(key_frame);
}

void Map::update_covisibility(std::shared_ptr<KeyFrame> key_frame)
{
  std::unordered_map<std::shared_ptr<KeyFrame>, size_t> key_frame_count;

  // for all the map points in the keyframe
  for (const auto mp: key_frame->get_map_points()) {

    // for all the key frames that see that map point
    for (auto kf: *mp) {
      auto shared_kf = kf.lock();
      if (shared_kf != key_frame) {
        if (key_frame_count.find(shared_kf) == key_frame_count.end()) {
          key_frame_count[shared_kf] = 1;
        } else {
          key_frame_count[shared_kf]++;
        }
      }
    }
  }

  for (const auto & [kf, count]: key_frame_count) {
    if (count > covisibility_minimum) {
      covisibility_insert(key_frame, kf);
    }
  }
}

void Map::covisibility_insert(
  std::shared_ptr<KeyFrame> key_frame_1,
  std::shared_ptr<KeyFrame> key_frame_2)
{
  covisibility[key_frame_1].push_back(key_frame_2);
  covisibility[key_frame_2].push_back(key_frame_1);
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
      ) == visited.end();
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

void Map::local_bundle_adjustment(std::shared_ptr<KeyFrame> key_frame, PinholeModel model)
{
  // get the key frames connected to the current key frame directly
  std::vector key_frames = {key_frame};
  for (const auto kf: covisibility[key_frame]) {
    key_frames.push_back(kf);
  }

  // get all the map points in the local map
  auto map_points = key_frames | std::views::transform(
    [](auto & kf) {
      return kf->get_map_points();
    }) | std::views::join;

  // restrict it to map points that are only seen by at least 2 views
  std::unordered_map<std::shared_ptr<MapPoint>, size_t> mp_count;
  for (const auto mp: map_points) {
    if (mp_count.find(mp) == mp_count.end()) {
      mp_count[mp] = 1;
    } else {
      mp_count[mp]++;
    }
  }

  std::unordered_set<std::shared_ptr<MapPoint>> mp_in_opt;
  for (const auto [mp, count]: mp_count) {
    if (count > 1) {
      mp_in_opt.insert(mp);
    }
  }

  // find the key frames that are connected to map points but are not
  // connected to the key frame in the covisibility graph
  std::unordered_set<std::shared_ptr<KeyFrame>> second_key_frames;
  for (const auto mp: mp_in_opt) {
    for (const auto kf: *mp) {
      if (std::find(key_frames.begin(), key_frames.end(), kf.lock()) == key_frames.end()) {
        second_key_frames.insert(kf.lock());
        key_frames.push_back(kf.lock());
      }
    }
  }


  // create axis angle representation of each keyframe rotation
  std::vector<std::array<double, 6>> camera_params;
  for (const auto kf: key_frames) {
    Eigen::AngleAxisd aa(kf->world_to_camera().block<3, 3>(0, 0));
    const auto rot = aa.axis() * aa.angle();
    camera_params.push_back(
      {
        rot[0],
        rot[1],
        rot[2],
        kf->world_to_camera()(0, 3),
        kf->world_to_camera()(1, 3),
        kf->world_to_camera()(2, 3)
      });
  }


  // create the bundle adjustment problem
  ceres::Problem problem;
  for (size_t idx = 0; idx < key_frames.size(); idx++) {
    for (const auto mp: key_frames[idx]->get_map_points()) {
      if (mp_in_opt.find(mp) == mp_in_opt.end()) {
        continue;
      }

      const auto [observed_x, observed_y] = key_frames[idx]->get_observed_location(mp);
      auto * cost_function = ReprojectionError::Create(observed_x, observed_y, model);

      problem.AddResidualBlock(
        cost_function, nullptr, camera_params[idx].data(),
        mp->position().data());
      if (key_frames[idx] == keyframes[0]) { // fix the first camera
        problem.SetParameterBlockConstant(camera_params[idx].data());
      }

      // fix the cameras in the second camera layer
      if (second_key_frames.find(key_frames[idx]) != second_key_frames.end()) {
        problem.SetParameterBlockConstant(camera_params[idx].data());
      }
    }
  }


  ceres::Solver::Options options;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  // options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  for (size_t idx = 0; idx < key_frames.size(); idx++) {
    Eigen::Vector3d translation = {
      camera_params[idx][3],
      camera_params[idx][4],
      camera_params[idx][5]
    };
    Eigen::Vector3d aa_vec = {
      camera_params[idx][0],
      camera_params[idx][1],
      camera_params[idx][2]
    };

    Eigen::Matrix3d rotation;
    rotation = Eigen::AngleAxisd(aa_vec.norm(), aa_vec.normalized());

    // create 4x4 transformation matrix
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = rotation;
    transform.block<3, 1>(0, 3) = translation;
    key_frames[idx]->set_world_to_camera(transform);
  }
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
