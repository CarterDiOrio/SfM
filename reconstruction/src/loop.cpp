#include "reconstruction/loop.hpp"
#include "reconstruction/graph_error_term.hpp"
#include <ranges>
#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <ceres/solver.h>
#include <ceres/types.h>

namespace sfm
{
LoopCloser::LoopCloser(
  std::shared_ptr<PlaceRecognition> place_recognition,
  std::shared_ptr<Map> map,
  std::shared_ptr<cv::DescriptorMatcher> matcher,
  LoopCloserOptions options)
: place_recognition{place_recognition}, map{map}, matcher{matcher}, options{options}
{}

void LoopCloser::detect_loops(KeyFramePtr key_frame)
{
  const auto candidate_keyframes = detect_candidate_keyframes(key_frame);
  auto candidate_groups = find_groups(key_frame, candidate_keyframes);
  auto loop = group_geometric_check(key_frame, candidate_groups);
  if (loop.has_value()) {
    close_loop(key_frame, loop.value());
  }
}

std::vector<KeyFramePtr> LoopCloser::detect_candidate_keyframes(KeyFramePtr key_frame)
{
  // get neighbors in the covisibility graph
  const auto neighbors = map->get_neighbors(key_frame, 30);

  // find the minimum score against the neighboring bow vectors
  double min_score = std::numeric_limits<double>::max();
  for (const auto & neighbor: neighbors) {
    const auto score = place_recognition->score(*key_frame, *neighbor);
    if (score < min_score) {
      min_score = score;
    }
  }

  // query with the minimum score amongst the neighbors
  const auto matches = place_recognition->query(*key_frame, -1, *map, min_score);

  // exclude all matches in the covisibility graph
  std::vector<std::shared_ptr<KeyFrame>> loop_candidates;
  std::copy_if(
    matches.begin(), matches.end(), std::back_inserter(loop_candidates),
    [&neighbors, &key_frame](const auto & match) {
      return std::find(neighbors.begin(), neighbors.end(), match) == neighbors.end();
    });

  return loop_candidates;
}


std::vector<LoopCloser::KeyFrameGroup> LoopCloser::find_groups(
  KeyFramePtr key_frame,
  const std::vector<KeyFramePtr> & candidates
)
{
  std::vector<bool> in_group(candidates.size());
  for (auto & group: key_frame_groups) {

    for (const auto & [idx, candidate]: std::views::enumerate(candidates)) {
      if (!in_group[idx]) {
        const auto & candidate_cov = map->get_neighbors(candidate, 60);
        if (group.covisibility.find(candidate) != group.covisibility.end()) {
          // is the candidate in the group directly
          group.expanded = true;
          group.expanded_count++;
          group.key_frames.insert(candidate);
          group.covisibility.insert(candidate_cov.begin(), candidate_cov.end());
          group.covisibility.insert(candidate);
          in_group.at(idx) = true;
          break;
        } else {
          // is the candidate in the group indirectly
          for (const auto & kf: candidate_cov) {
            if (group.covisibility.find(kf) != group.covisibility.end()) {
              group.expanded = true;
              group.expanded_count++;
              group.key_frames.insert(candidate);
              group.covisibility.insert(candidate_cov.begin(), candidate_cov.end());
              group.covisibility.insert(candidate);
              in_group.at(idx) = true;
              break; // only need to find one
            }
          }
        }
      }

      if (group.expanded) {
        break; // only expand each group once
      }
    }
  }

  // // remove groups that were not expanded
  key_frame_groups.erase(
    std::remove_if(
      key_frame_groups.begin(), key_frame_groups.end(),
      [](auto & group) {
        const auto remove = !group.expanded;
        group.expanded = false; // reset for next iteration
        return remove;
      }
    ),
    key_frame_groups.end()
  );


  // // check if any groups meet the threshold
  std::vector<LoopCloser::KeyFrameGroup> consistent_groups;
  std::copy_if(
    key_frame_groups.begin(), key_frame_groups.end(), std::back_inserter(consistent_groups),
    [](const auto & group) {
      return group.expanded_count >= expansion_consistency_threshold;
    });

  // // remove groups that meet the expansion threshold
  key_frame_groups.erase(
    std::remove_if(
      key_frame_groups.begin(), key_frame_groups.end(),
      [](const auto & group) {
        return group.expanded_count >= expansion_consistency_threshold;
      }
    ),
    key_frame_groups.end()
  );

  // create new groups for the candidates that are not in a group
  for (const auto & [idx, in]: std::views::enumerate(in_group)) {
    if (!in) {
      KeyFrameGroup group;
      group.key_frames.insert(candidates[idx]);
      group.covisibility.insert(candidates[idx]);
      const auto & neighbors = map->get_neighbors(candidates[idx], 30);
      group.covisibility.insert(neighbors.begin(), neighbors.end());
      key_frame_groups.push_back(group);
    }
  }

  return consistent_groups;
}

std::optional<LoopCloser::KeyFrameGroup> LoopCloser::group_geometric_check(
  KeyFramePtr key_frame,
  std::vector<LoopCloser::KeyFrameGroup> & groups
)
{
  for (auto & group: groups) {
    // get all the map points in the group
    for (const auto & kf: group.key_frames) {
      const auto mps = kf->get_map_points();
      group.map_points.insert(mps.begin(), mps.end());
    }

    std::vector<std::shared_ptr<MapPoint>> map_points{group.map_points.begin(),
      group.map_points.end()};

    // create a cv::Mat of the descriptors for all the map points
    cv::Mat descriptors;
    for (const auto & mp: map_points) {
      descriptors.push_back(mp->description());
    }
    std::vector<cv::DMatch> matches;
    matcher->match(key_frame->get_descriptors_mat(), descriptors, matches);

    // get 2D-3D correspondences
    std::vector<cv::Point2d> image_points;
    std::vector<Eigen::Vector3d> world_points;
    for (const auto & match: matches) {
      image_points.push_back(key_frame->get_keypoints()[match.queryIdx].pt);
      world_points.push_back(map_points[match.trainIdx]->position());
    }

    try {
      const auto [transformation, inliers] = pnp(
        image_points, world_points,
        key_frame->camera_calibration());

      size_t count = std::count_if(
        inliers.begin(), inliers.end(), [](const auto & inlier) {
          return inlier > 0;
        });

      if (count > 30) {
        group.T_wk = transformation;
        return group;
      }

    } catch (cv::Exception & e) {
      // std::cout << e.what() << std::endl;
    }
  }

  return {};
}

void LoopCloser::close_loop(KeyFramePtr key_frame, KeyFrameGroup & group)
{
  std::cout << "loop closure" << std::endl;
  auto loop_key_frames = map->get_neighbors(key_frame);

  // the loop closed transform propogated to each of the key frames neighbors
  Eigen::Matrix4d original_transform = key_frame->world_to_camera();
  Eigen::Matrix4d corrected_transform = group.T_wk.inverse();
  key_frame->set_world_to_camera(corrected_transform);

  std::set<std::shared_ptr<MapPoint>> group_map_points;
  for (const auto & kf: group.key_frames) {
    const auto mps = kf->get_map_points();
    group_map_points.insert(mps.begin(), mps.end());
    for (const auto & neigh: map->get_neighbors(kf)) {
      const auto mps = neigh->get_map_points();
      group_map_points.insert(mps.begin(), mps.end());
    }
  }


  auto visibile_map_points = group_map_points | std::views::filter(
    [&key_frame](const auto & mp) {
      return key_frame->point_in_frame(*mp);
    });

  for (const auto & mp: visibile_map_points) {
    auto projection = project_map_point(*key_frame, *mp);

    const auto features = key_frame->get_features_within_radius(
      projection.x,
      projection.y,
      2.0,
      true);

    if (features.size() > 0) {
      const auto mp_desc = mp->description();

      // find the minimum descriptor distance
      auto min_idx = *std::min_element(
        features.begin(), features.end(),
        [&mp_desc, &key_frame](const auto & idx1, const auto & idx2) {
          return cv::norm(mp_desc, key_frame->get_point(idx1).second, cv::NORM_HAMMING) <
                 cv::norm(mp_desc, key_frame->get_point(idx2).second, cv::NORM_HAMMING);
        });

      const auto kf_mp = key_frame->corresponding_map_point(min_idx);
      if (kf_mp.has_value()) {
        map->unlink_kf_and_mp(key_frame, kf_mp.value());
      }

      map->link_keyframe_to_map_point(key_frame, min_idx, mp);
    }
  }

  // add edges in the covisibility graph
  const auto neighbors_before = map->get_neighbors(key_frame);
  map->update_covisibility(key_frame);
  const auto neighbors_after = map->get_neighbors(key_frame);

  std::vector<std::pair<KeyFramePtr, Eigen::Matrix4d>> loop_edges;
  std::unordered_set<KeyFramePtr> loop_edge_kfs;

  // identify new loop edges
  for (const auto & new_neighbor: neighbors_after) {
    if (std::find(neighbors_before.begin(), neighbors_before.end(), new_neighbor) ==
      neighbors_before.end())
    {
      loop_edges.push_back(
        {new_neighbor, key_frame->world_to_camera() * new_neighbor->transform()});
      loop_edge_kfs.insert(new_neighbor);
    }
  }

  // create se3 representations for each frame
  std::unordered_map<KeyFramePtr, Eigen::Vector<double, 6>> kf_poses;
  for (const auto & kf: map->get_key_frames()) {
    Sophus::SE3<double> SE3{kf->world_to_camera()};
    Eigen::Matrix<double, 6, 1> se3_vec = SE3.log();
    kf_poses[kf] = se3_vec;
  }

  // optimize
  ceres::Problem problem;
  ceres::LossFunction * loss_function = nullptr;

  // add loop edges to the problem
  for (const auto & [kf_b, T_ab]: loop_edges) {
    auto & a_se3_vec = kf_poses[key_frame];
    auto & b_se3_vec = kf_poses[kf_b];
    ceres::CostFunction * cost_function = PoseGraph3dErrorTerm::Create(T_ab);
    problem.AddResidualBlock(
      cost_function, loss_function,
      a_se3_vec.data(), b_se3_vec.data());
  }

  // get all normal edges to the problem
  std::vector<std::pair<KeyFramePtr, KeyFramePtr>> normal_edges;
  for (const auto [pair, count]: map->covisibility_edge) {
    if (count >= 30) {
      if (pair.first != key_frame && pair.second != key_frame) {
        //neither are loop key frame, can't be a loop edge
        normal_edges.push_back(pair);
      } else if (pair.first == key_frame &&
        loop_edge_kfs.find(pair.second) == loop_edge_kfs.end())
      {
        // kf is the key frame and nkf is not a loop edge
        normal_edges.push_back(pair);
      } else if (pair.second == key_frame &&
        loop_edge_kfs.find(pair.first) == loop_edge_kfs.end())
      {
        // nkf is the key frame and kf is not a loop edge
        normal_edges.push_back(pair);
      }
    }
  }

  // add normal edges to the problem
  for (const auto & [kf_a, kf_b]: normal_edges) {
    auto & a_pose = kf_poses[kf_a];
    auto & b_pose = kf_poses[kf_b];

    Eigen::Matrix4d T_ab;
    if (kf_a == key_frame) {
      T_ab = original_transform * kf_b->transform();
    } else if (kf_b == key_frame) {
      T_ab = kf_a->world_to_camera() * Sophus::SE3<double>{original_transform}.inverse().matrix();
    } else {
      T_ab = kf_a->world_to_camera() * kf_b->transform();
    }

    ceres::CostFunction * cost_function = PoseGraph3dErrorTerm::Create(
      T_ab);
    problem.AddResidualBlock(
      cost_function, loss_function,
      a_pose.data(), b_pose.data());
  }

  // set the loop key frame as constant
  const auto & se3_vec = kf_poses[key_frame];
  problem.SetParameterBlockConstant(se3_vec.data());

  // also lock the original key frame to prevent the map from shifting
  const auto & original_se3_vec = kf_poses[map->keyframes[0]];
  problem.SetParameterBlockConstant(original_se3_vec.data());

  ceres::Solver::Options options;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.dogleg_type = ceres::SUBSPACE_DOGLEG;
  ceres::Solver::Summary summary;
  options.max_num_iterations = 50;
  ceres::Solve(options, &problem, &summary);

  for (const auto & [kf, se3_vec]: kf_poses) {
    Eigen::Matrix4d tf = Sophus::SE3<double>::exp(se3_vec).matrix();
    kf->set_world_to_camera(tf);
  }

  // update all the map points
  for (const auto & mp: map->mappoints) {
    mp->update_position();
  }

  std::cout << "loop closed" << std::endl;
  // clear key frame groups
  key_frame_groups.clear();
}

}
