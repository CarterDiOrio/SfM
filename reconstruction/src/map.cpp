#include "reconstruction/map.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/AngleAxis.h>
#include <algorithm>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/types.h>
#include <cmath>
#include <opencv2/core.hpp>
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
#include <sophus/se3.hpp>

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
  cv::Mat depth,
  PinholeModel model)
{
  const auto new_keyframe =
    std::make_shared<KeyFrame>(K, T_kw, keypoints, descriptions, img, depth, model);
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
  if (key_frame->link_map_point(key_point_idx, map_point)) {
    map_point->add_keyframe(key_frame);
  }
}

void Map::unlink_kf_and_mp(std::shared_ptr<KeyFrame> key_frame, std::shared_ptr<MapPoint> mp)
{
  key_frame->remove_map_point(mp);
  mp->remove_keyframe(key_frame);

  // if no more key frames reference the map point remove the map point
  if (mp->get_keyframes().size() == 0) {
    remove_map_point(mp);
  }
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
    covisibility_insert(key_frame, kf, count);
  }
}

void Map::covisibility_insert(
  std::shared_ptr<KeyFrame> key_frame_1,
  std::shared_ptr<KeyFrame> key_frame_2,
  size_t count)
{
  // check if the key frames are already in the graph
  if (covisibility_edge.find(edge_order(key_frame_1, key_frame_2)) != covisibility_edge.end()) {
    return;
  }

  insert_edge(key_frame_1, key_frame_2, count);
}

std::unordered_set<KeyFramePtr> Map::get_local_map(
  std::shared_ptr<KeyFrame> key_frame,
  size_t distance,
  size_t min_shared_features
)
{
  std::unordered_set<KeyFramePtr> local_map;

  size_t current_distance = 1;
  std::deque<KeyFramePtr> queue;
  queue.push_back(key_frame);
  while (!queue.empty() && current_distance <= distance) {

    size_t layer_size = queue.size();
    for (size_t i = 0; i < layer_size; i++) {
      const auto current_key_frame = queue.front();
      const auto neighbors = get_neighbors(current_key_frame, 1);
      for (const auto kf: neighbors) {
        if (local_map.find(kf) == local_map.end()) {
          local_map.insert(kf);
          queue.push_back(kf);
        }
      }
    }

    current_distance++;
  }

  return local_map;
}

std::vector<std::shared_ptr<KeyFrame>> Map::get_neighbors(
  std::shared_ptr<KeyFrame> key_frame, size_t min_shared_features)
{
  auto kfs = covisibility[key_frame] | std::views::filter(
    [&key_frame, min_shared_features, & edges = covisibility_edge, this](const auto & kf) {
      return this->get_edge(key_frame, kf) > min_shared_features;
    });
  return {kfs.begin(), kfs.end()};
}

void Map::remove_map_point(std::shared_ptr<MapPoint> map_point)
{
  mappoints.erase(
    std::remove(mappoints.begin(), mappoints.end(), map_point),
    mappoints.end());
  const auto keyframes = map_point->get_keyframes();
  for (const auto kf: keyframes) {
    kf.lock()->remove_map_point(map_point);
  }
}

std::vector<KeyFramePtr> Map::get_key_frames()
{
  return keyframes;
}

bool Map::check_keyframe_redundancy(
  std::shared_ptr<KeyFrame> key_frame,
  double threshold)
{
  const auto map_points = key_frame->get_map_points();
  size_t count = 0;

  for (const auto mp: map_points) {

    // check if the map point has been seen by 3 other frames at the same or
    // finer scale
    if (mp->get_keyframes().size() > 1) {

      // reference distance
      const auto ref_dist = (key_frame->world_to_camera() * mp->position().homogeneous()).norm();

      size_t scale_count = 0;
      for (const auto kf: mp->get_keyframes()) {
        if (kf.lock() != key_frame) {
          // get distance to map point from kf
          const auto dist = (kf.lock()->world_to_camera() * mp->position().homogeneous()).norm();

          // get distance between the two keyframes
          const auto dist_kf = (kf.lock()->world_to_camera().block<3, 1>(0, 3) -
            key_frame->world_to_camera().block<3, 1>(0, 3)).norm();

          if (dist <= ref_dist && dist_kf <= 100) {
            scale_count++;
          }
        }
      }

      if (scale_count >= 1) {
        count++;
      }
    }
  }

  const auto ratio = static_cast<double>(count) / map_points.size();
  return ratio >= threshold;
}

void Map::remove_key_frame(std::shared_ptr<KeyFrame> key_frame)
{
  const auto kf_mp = key_frame->get_map_points();

  // unlink map points from the key frame
  for (const auto mp: kf_mp) {
    mp->remove_keyframe(key_frame);
    if (mp->get_keyframes().size() == 0) {
      remove_map_point(mp);
    }
  }

  // unlink it from the covisibility graph
  const auto neighbors = covisibility[key_frame];
  for (const auto n_kf: neighbors) {
    auto edge = edge_order(key_frame, n_kf);
    covisibility_edge.erase(edge);
  }
  covisibility.erase(key_frame);

  // remove it from the key frames vector
  const auto it = std::find(keyframes.begin(), keyframes.end(), key_frame);
  keyframes.erase(it);
}

void Map::local_bundle_adjustment(PinholeModel model)
{
  // std::unordered_set<KeyFramePtr> ba_key_frames {local_map.begin(), local_map.end()};
  // ba_key_frames.insert(key_frame);

  std::unordered_set<KeyFramePtr> ba_key_frames = {keyframes.begin(), keyframes.end()};

  // get all the map points in the local map
  std::unordered_set<MapPointPtr> map_points;
  for (const auto kf: ba_key_frames) {
    for (const auto mp: kf->get_map_points()) {
      if (mp->is_invalid()) {
        remove_map_point(mp);
      } else if (mp->get_keyframes().size() > 2) {
        map_points.insert(mp);
      }
    }
  }

  std::cout << "Map points: " << map_points.size() << "\n";

  // create se3 representation of each keyframe
  std::unordered_map<KeyFramePtr, Eigen::Vector<double, 6>> kf_poses;
  for (const auto & kf: ba_key_frames) {
    Sophus::SE3<double> SE3{kf->world_to_camera()};
    Eigen::Matrix<double, 6, 1> se3_vec = SE3.log();
    kf_poses[kf] = se3_vec;
  }

  std::unordered_map<MapPointPtr, Eigen::Vector3d> points;
  for (const auto & mp: map_points) {
    points[mp] = mp->position();
  }

  // create the bundle adjustment problem
  ceres::Problem problem;
  ceres::LossFunction * loss_function = new ceres::HuberLoss(0.1);
  for (const auto & kf: ba_key_frames) {
    for (const auto & mp: kf->get_map_points()) {

      if (map_points.find(mp) == map_points.end()) {
        continue;
      }

      const auto [observed_x, observed_y] = kf->get_observed_location(mp);
      auto * cost_function = ReprojectionError::Create(observed_x, observed_y, model);

      problem.AddResidualBlock(
        cost_function, loss_function, kf_poses[kf].data(),
        points[mp].data());

      if (kf == keyframes[0]) {  // fix the first key frame
        problem.SetParameterBlockConstant(kf_poses[kf].data());
      }
    }
  }

  ceres::Solver::Options options;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.minimizer_progress_to_stdout = true;
  options.use_explicit_schur_complement = true;
  options.max_linear_solver_iterations = 500;
  options.max_num_iterations = 500;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n";

  if (!summary.IsSolutionUsable()) {
    std::cerr << "Bundle adjustment failed\n";
    return;
  }

  // convert back to SE3
  for (const auto & [kf, se3_vec]: kf_poses) {
    Eigen::Matrix4d tf = Sophus::SE3<double>::exp(se3_vec).matrix();
    kf->set_world_to_camera(tf);
  }

  // update the map points
  for (const auto & mp: mappoints) {
    mp->update_position();
  }

  for (const auto & mp: map_points) {
    mp->set_position(points[mp]);
  }

}

size_t Map::get_edge(
  std::shared_ptr<KeyFrame> key_frame_1,
  std::shared_ptr<KeyFrame> key_frame_2)
{
  return covisibility_edge[edge_order(key_frame_1, key_frame_2)];
}

void Map::insert_edge(
  std::shared_ptr<KeyFrame> key_frame_1,
  std::shared_ptr<KeyFrame> key_frame_2,
  size_t count)
{
  covisibility[key_frame_1].push_back(key_frame_2);
  covisibility[key_frame_2].push_back(key_frame_1);
  covisibility_edge[edge_order(key_frame_1, key_frame_2)] = count;
}

std::pair<KeyFramePtr, KeyFramePtr> Map::edge_order(
  std::shared_ptr<KeyFrame> key_frame_1,
  std::shared_ptr<KeyFrame> key_frame_2)
{
  if (key_frame_1 < key_frame_2) {
    return {key_frame_1, key_frame_2};
  } else {
    return {key_frame_2, key_frame_1};
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
