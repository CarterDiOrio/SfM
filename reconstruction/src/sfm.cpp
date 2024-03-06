#include "reconstruction/sfm.hpp"

#include <Eigen/src/Geometry/Quaternion.h>
#include <algorithm>
#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <ceres/solver.h>
#include <ceres/types.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <sophus/se3.hpp>

#include "reconstruction/mappoint.hpp"
#include "reconstruction/keyframe.hpp"
#include "reconstruction/features.hpp"
#include "reconstruction/place_recognition.hpp"
#include "reconstruction/utils.hpp"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <ranges>
#include <time.h>

#include "reconstruction/loop.hpp"

namespace sfm
{
Reconstruction::Reconstruction(ReconstructionOptions options)
: matcher{cv::BFMatcher::create(cv::NORM_HAMMING, true)},
  detector{cv::ORB::create(4000
    )},
  model{options.model},
  options{options},
  place_recognition(options.place_recognition_voc)
{}

void Reconstruction::add_frame_ordered(const cv::Mat & frame, const cv::Mat & depth)
{
  // If no prior frames, initialize map with first frame as the origin:
  if (map.is_empty()) {
    initialize_reconstruction(frame, depth);
    return;
  }
  track_previous_frame(frame, depth);
}

void Reconstruction::initialize_reconstruction(const cv::Mat & frame, const cv::Mat & depth)
{
  // 1. Detect feautres in image
  const auto features = detect_features(frame, detector);
  const auto [keypoints, descriptors] = filter_features(features, depth, options.max_depth);

  // 2. project points to 3d
  auto points = deproject_keypoints(keypoints, depth, model);

  // 3. Create map points
  const auto initial = map.create_keyframe(
    model,
    Eigen::Matrix4d::Identity(),
    keypoints,
    descriptors,
    frame,
    depth,
    model
  );

  // 5. create and link new map points
  const auto shared_keyframe = initial.lock();
  for (const auto & [idx, map_point]: shared_keyframe->create_map_points()) {
    map.add_map_point(map_point);
  }

  // 6. set the bow vector for the place recognition
  shared_keyframe->set_bow_vector(place_recognition.convert(shared_keyframe->get_descriptors()));
  place_recognition.add(shared_keyframe);

  previous_keyframe = initial;
}

void Reconstruction::track_previous_frame(const cv::Mat & frame, const cv::Mat & depth)
{
  auto first = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch());

  // 1. get orb features in frame
  const auto features = detect_features(frame, detector);
  const auto [keypoints, descriptors] = filter_features(features, depth, options.max_depth);

  const auto colors = extract_colors(frame, keypoints);

  //2. find matches to previous frame map points
  const auto previous_shared = previous_keyframe.lock();
  const auto matches = previous_shared->match(descriptors, matcher);

  // 2.1 for each match get the map points
  std::vector<std::pair<size_t, std::shared_ptr<MapPoint>>> kp_mp_pairs;
  for (const auto & dmatch: matches) {
    const auto mp = previous_shared->corresponding_map_point(dmatch.trainIdx);
    if (mp.has_value()) {
      kp_mp_pairs.emplace_back(static_cast<size_t>(dmatch.queryIdx), mp.value());
    }
  }

  // draw matches between frames
  cv::Mat mimg;
  cv::drawMatches(
    frame, keypoints, previous_shared->img,
    previous_shared->get_keypoints(), matches, mimg);
  cv::imshow("matches", mimg);


  // 3. Compute PnP from map points to 2D locations in current frame
  std::vector<cv::Point2d> image_points;
  std::vector<Eigen::Vector3d> world_points;
  for (const auto & pair: kp_mp_pairs) {
    image_points.push_back(keypoints[pair.first].pt);
    world_points.push_back(pair.second->position());
  }

  auto [transformation, inliers] = pnp(image_points, world_points);

  // filter for pnp inliers
  std::vector<std::pair<size_t, std::shared_ptr<MapPoint>>> filtered_kp_mp;
  for (const auto [idx, inlier_val]: std::views::enumerate(inliers)) {
    if (inlier_val > 0) {
      filtered_kp_mp.push_back(kp_mp_pairs[idx]);
    }
  }

  //4. create key frame
  const auto current = map.create_keyframe(
    model,
    transformation,
    keypoints,
    descriptors,
    frame,
    depth,
    model
  );
  const auto shared_current = current.lock();

  // add to place recognition engine
  shared_current->set_bow_vector(place_recognition.convert(shared_current->get_descriptors()));
  place_recognition.add(shared_current);

  //5. add matched map points to keyframe
  for (const auto & pair: filtered_kp_mp) { // only link the key points that were inliers in pnp
    map.link_keyframe_to_map_point(shared_current, pair.first, pair.second);
  }
  map.update_covisibility(shared_current); // update the covisibility graph

  // track the local map
  track_local_map(shared_current);

  //perform pnp
  image_points.clear();
  world_points.clear();
  for (const auto & mp: shared_current->get_map_points()) {
    const auto pt = shared_current->get_observed_location(mp);
    image_points.push_back(cv::Point2d{pt.first, pt.second});
    world_points.push_back(mp->position());
  }

  const auto [local_transformation, local_inliers] = pnp(image_points, world_points);
  shared_current->set_world_to_camera(local_transformation.inverse());
  for (const auto [idx, mp]: std::views::enumerate(shared_current->get_map_points())) {
    // remove outliers
    if (local_inliers[idx] < 1) {
      map.unlink_kf_and_mp(shared_current, mp);
    }
  }
  map.update_covisibility(shared_current);


  // create new map points from unmatched points
  const auto mps = shared_current->create_map_points();
  for (const auto & [idx, map_point]: mps) {
    map.add_map_point(map_point);
  }
  map.update_covisibility(shared_current);

  // attempt to loop close
  loop_closing(shared_current);

  // check if local map needs to be pruned
  for (const auto & kf: map.get_neighbors(shared_current)) {
    const auto is_redundant = map.check_keyframe_redundancy(kf);
    if (is_redundant) {
      map.remove_key_frame(kf);
      place_recognition.forbid(kf);
    }
  }


  auto second = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch());

  // std::cout << "time: " << second.count() - first.count() << std::endl;

  previous_keyframe = current;
}

void Reconstruction::track_local_map(std::shared_ptr<KeyFrame> key_frame)
{
  const auto local_map = map.get_local_map(key_frame, 2, 15);

  // map points in current keyframe
  auto current_map_points = key_frame->get_map_points();
  std::unordered_set<std::shared_ptr<MapPoint>> current_set{current_map_points.begin(),
    current_map_points.end()};

  // get all the map points in the local map not in the current key frame
  auto mps = local_map |
    std::views::transform([](auto & kf) {return kf->get_map_points();}) |
    std::views::join |
    std::views::filter(
    [&current_set](auto & mp) {
      return current_set.find(mp) == current_set.end();
    }) |
    std::views::filter(
    [&key_frame](auto & mp) {
      return key_frame->point_in_frame(*mp);
    });

  std::unordered_set<std::shared_ptr<MapPoint>> local_set;
  for (const auto mp: mps) {
    local_set.insert(mp);
  }

  std::cout << "local map size: " << local_map.size() << " total size: " <<
    map.get_key_frames().size() <<
    std::endl;


  // filter map points
  for (auto mp: local_set) {
    auto projection = project_map_point(*key_frame, *mp);
    const auto features = key_frame->get_features_within_radius(
      projection.x, projection.y, 2.0,
      false);

    if (features.size() > 0) {
      const auto mp_desc = mp->description();

      // get the features with the minimum distance
      double min_dist =
        cv::norm(mp_desc, key_frame->get_point(features[0]).second, cv::NORM_HAMMING);
      size_t min_idx = features[0];

      for (size_t idx: features) {
        double dist =
          cv::norm(mp_desc, key_frame->get_point(idx).second, cv::NORM_HAMMING);
        if (dist < min_dist) {
          min_idx = idx;
          min_dist = dist;
        }
      }

      // link the matched point to the key frame
      map.link_keyframe_to_map_point(key_frame, min_idx, mp);
    }
  }

}

std::pair<Eigen::Matrix4d, std::vector<int>> Reconstruction::pnp(
  const std::vector<cv::Point2d> & image_points,
  const std::vector<Eigen::Vector3d> & world_points)
{
  std::vector<cv::Point3d> world_points_cv(world_points.size());
  std::transform(
    world_points.begin(), world_points.end(), world_points_cv.begin(),
    [](const Eigen::Vector3d & vec) {
      return cv::Point3d{vec(0), vec(1), vec(2)};
    }
  );

  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);

  std::vector<int> inliers;
  cv::Mat k = model_to_mat(model);
  cv::solvePnPRansac(
    world_points_cv, image_points, k,
    cv::noArray(), rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_SQPNP);

  // convert screw to transformation matrix
  cv::Mat rotation_mat;
  cv::Rodrigues(rvec, rotation_mat);

  Eigen::Matrix3d rotation_eig;
  cv::cv2eigen(rotation_mat, rotation_eig);
  Eigen::Vector3d translation_eig{tvec.at<double>(0, 0), tvec.at<double>(1, 0),
    tvec.at<double>(2, 0)};

  // the transformation is given from world to camera and we want camera to world
  Eigen::Matrix4d transformation;
  transformation.setIdentity();
  transformation.block<3, 3>(0, 0) = rotation_eig.transpose();
  transformation.block<3, 1>(0, 3) = -1 * rotation_eig.transpose() * translation_eig;
  return {transformation, inliers};
}

void Reconstruction::loop_closing(std::shared_ptr<KeyFrame> key_frame)
{
  const auto candidate_keyframes = loop_candidate_detection(key_frame);
  auto candidate_groups = loop_candidate_refinment(key_frame, candidate_keyframes);
  auto loop = loop_candidate_geometric(key_frame, candidate_groups);
  if (loop.has_value()) {
    loop_closure(key_frame, loop.value());
  }
}

std::vector<std::shared_ptr<KeyFrame>> Reconstruction::loop_candidate_detection(
  std::shared_ptr<KeyFrame> key_frame)
{
  // get neighbors in the covisibility graph
  const auto neighbors = map.get_neighbors(key_frame, 30);

  // find the minimum score against the neighboring bow vectors
  double min_score = std::numeric_limits<double>::max();
  for (const auto & neighbor: neighbors) {
    const auto score = place_recognition.score(*key_frame, *neighbor);
    if (score < min_score) {
      min_score = score;
    }
  }
  // query with the minimum score amongst the neighbors
  const auto matches = place_recognition.query(*key_frame, -1, map, min_score);

  // exclude all matches in the covisibility graph
  std::vector<std::shared_ptr<KeyFrame>> loop_candidates;
  std::copy_if(
    matches.begin(), matches.end(), std::back_inserter(loop_candidates),
    [&neighbors, &key_frame](const auto & match) {
      return std::find(neighbors.begin(), neighbors.end(), match) == neighbors.end();
    });

  return loop_candidates;
}

std::vector<KeyFrameGroup> Reconstruction::loop_candidate_refinment(
  const std::shared_ptr<KeyFrame> key_frame,
  const std::vector<std::shared_ptr<KeyFrame>> & candidates)
{
  std::vector<bool> in_group(candidates.size());
  for (auto & group: keyframe_groups) {

    for (const auto & [idx, candidate]: std::views::enumerate(candidates)) {
      if (!in_group[idx]) {
        const auto & candidate_cov = map.get_neighbors(candidate, 60);
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

  // remove groups that were not expanded
  keyframe_groups.erase(
    std::remove_if(
      keyframe_groups.begin(), keyframe_groups.end(),
      [](auto & group) {
        const auto remove = !group.expanded;
        group.expanded = false; // reset for next iteration
        return remove;
      }
    ),
    keyframe_groups.end()
  );


  // check if any groups meet the threshold
  std::vector<KeyFrameGroup> consistent_groups;
  std::copy_if(
    keyframe_groups.begin(), keyframe_groups.end(), std::back_inserter(consistent_groups),
    [](const auto & group) {
      return group.expanded_count >= expansion_consistency_threshold;
    });

  // remove groups that meet the expansion threshold
  keyframe_groups.erase(
    std::remove_if(
      keyframe_groups.begin(), keyframe_groups.end(),
      [](const auto & group) {
        return group.expanded_count >= expansion_consistency_threshold;
      }
    ),
    keyframe_groups.end()
  );

  // create new groups for the candidates that are not in a group
  for (const auto & [idx, in]: std::views::enumerate(in_group)) {
    if (!in) {
      KeyFrameGroup group;
      group.key_frames.insert(candidates[idx]);
      group.covisibility.insert(candidates[idx]);
      const auto & neighbors = map.get_neighbors(candidates[idx], 30);
      group.covisibility.insert(neighbors.begin(), neighbors.end());
      keyframe_groups.push_back(group);
    }
  }

  return consistent_groups;
}

std::optional<KeyFrameGroup> Reconstruction::loop_candidate_geometric(
  std::shared_ptr<KeyFrame> key_frame,
  std::vector<KeyFrameGroup> & groups)
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
      const auto [transformation, inliers] = pnp(image_points, world_points);

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

void Reconstruction::loop_closure(std::shared_ptr<KeyFrame> key_frame, KeyFrameGroup & group)
{
  std::cout << "loop closure" << std::endl;
  auto loop_key_frames = map.get_neighbors(key_frame);

  // the loop closed transform propogated to each of the key frames neighbors
  Eigen::Matrix4d original_transform = key_frame->world_to_camera();
  Eigen::Matrix4d corrected_transform = group.T_wk.inverse();
  key_frame->set_world_to_camera(corrected_transform);

  std::set<std::shared_ptr<MapPoint>> group_map_points;
  for (const auto & kf: group.key_frames) {
    const auto mps = kf->get_map_points();
    group_map_points.insert(mps.begin(), mps.end());
    for (const auto & neigh: map.get_neighbors(kf)) {
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
        map.unlink_kf_and_mp(key_frame, kf_mp.value());
      }

      map.link_keyframe_to_map_point(key_frame, min_idx, mp);
    }
  }

  // add edges in the covisibility graph
  const auto neighbors_before = map.get_neighbors(key_frame);
  map.update_covisibility(key_frame);
  const auto neighbors_after = map.get_neighbors(key_frame);

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
  for (const auto & kf: map.get_key_frames()) {
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
  for (const auto [pair, count]: map.covisibility_edge) {
    if (count >= 100) {
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
  const auto & original_se3_vec = kf_poses[map.keyframes[0]];
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
  for (const auto & mp: map.mappoints) {
    mp->update_position();
  }

  std::cout << "loop closed" << std::endl;
  // clear key frame groups
  keyframe_groups.clear();
}

}
