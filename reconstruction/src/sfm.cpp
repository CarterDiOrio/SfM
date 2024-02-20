#include "reconstruction/sfm.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <ctime>
#include <iterator>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "reconstruction/mappoint.hpp"
#include "reconstruction/keyframe.hpp"
#include "reconstruction/features.hpp"
#include "reconstruction/place_recognition.hpp"
#include "reconstruction/utils.hpp"

#include <unordered_set>
#include <vector>
#include <ranges>
#include <time.h>

#include "reconstruction/loop.hpp"

namespace sfm
{
Reconstruction::Reconstruction(ReconstructionOptions options)
: matcher{cv::BFMatcher::create(cv::NORM_HAMMING, true)},
  detector{cv::ORB::create(2000)},
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
  auto start = std::clock();

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

  shared_current->set_bow_vector(place_recognition.convert(shared_current->get_descriptors()));
  place_recognition.add(shared_current);

  //5. add matched map points to keyframe
  for (const auto & pair: filtered_kp_mp) { // only link the key points that were inliers in pnp
    map.link_keyframe_to_map_point(shared_current, pair.first, pair.second);
  }
  map.update_covisibility(shared_current); // update the covisibility graph

  // track the local map
  track_local_map(shared_current);

  // create new map points from unmatched points
  const auto mps = shared_current->create_map_points();
  for (const auto & [idx, map_point]: mps) {
    map.add_map_point(map_point);
  }
  map.update_covisibility(shared_current);

  // attempt to loop close
  loop_closing(shared_current);

  std::cout << shared_current->transform().row(0) << ' ' << shared_current->transform().row(1) <<
    ' ' <<
    shared_current->transform().row(2) << std::endl;


  previous_keyframe = current;
}

void Reconstruction::track_local_map(std::shared_ptr<KeyFrame> key_frame)
{
  const auto local_map = map.get_local_map(key_frame, 2);

  // map points in current keyframe
  auto current_map_points = key_frame->get_map_points();
  std::unordered_set<std::shared_ptr<MapPoint>> current_set{current_map_points.begin(),
    current_map_points.end()};

  // filter for points in the view of the camera (in front and in frame)
  const auto in_front_filter = [transform = key_frame->world_to_camera()](const auto mp) {
      const auto img_p = transform * mp->position().homogeneous();
      return img_p.z() > 0 && img_p.x() >= 0 && img_p.x() <= 1280 && img_p.y() >= 0 &&
             img_p.y() <= 720;
    };

  // get all the map points in the local map
  auto mps = local_map |
    std::views::join |
    std::views::transform([](auto & kf) {return kf->get_map_points();}) |
    std::views::join |
    std::views::filter(
    [&current_set](auto & mp) {
      return current_set.find(mp) == current_set.end();
    }) |
    std::views::filter(in_front_filter);

  std::unordered_set<std::shared_ptr<MapPoint>> local_set;
  for (const auto mp: mps) {
    local_set.insert(mp);
  }


  // filter map points
  for (auto mp: local_set) {
    auto projection = project_map_point(*key_frame, *mp);
    const auto features = key_frame->get_features_within_radius(projection.x, projection.y, 3.0);

    if (features.size() > 0) {
      const auto mp_desc = mp->description();

      // get the features with the minimum distance
      double min_dist = cv::norm(mp_desc, key_frame->get_point(0).second, cv::NORM_HAMMING);
      size_t min_idx = features[0];

      for (size_t idx: features) {
        double dist = cv::norm(mp_desc, key_frame->get_point(0).second, cv::NORM_HAMMING);
        if (dist < min_dist) {
          min_idx = idx;
          min_dist = dist;
        }
      }

      // link the matched point to the key frame
      map.link_keyframe_to_map_point(key_frame, min_idx, mp);
    }
  }

  map.update_covisibility(key_frame);
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
        const auto & candidate_cov = map.get_neighbors(candidate, 30);
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

    // get 2D 3D correspondences
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
        // std::cout << "inliers: " << count << std::endl;
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
  std::cout << "LOOP CLOSURE!" << std::endl;
  //get the transform between the key frame and each of its neighbors
  auto loop_key_frames = map.get_neighbors(key_frame);
  const Eigen::Matrix4d T_kw = group.T_wk.inverse();

  // the loop closed transform propogated to each of the key frames neighbors
  for (auto & neighbor: loop_key_frames) {
    // We want T_kn_w = T_kn_k  * T_kw
    // T_kn_k = T_kn_w * T_wk
    Eigen::Matrix4d T_kn_k = neighbor->world_to_camera() * key_frame->transform();
    Eigen::Matrix4d T_kn_w = T_kn_k * T_kw;
    neighbor->set_world_to_camera(T_kn_w);
  }
  key_frame->set_world_to_camera(T_kw);
  loop_key_frames.push_back(key_frame);

  std::set<std::shared_ptr<MapPoint>> group_map_points;
  for (const auto & kf: group.key_frames) {
    const auto mps = kf->get_map_points();
    group_map_points.insert(mps.begin(), mps.end());
    for (const auto & neigh: map.get_neighbors(kf)) {
      const auto mps = neigh->get_map_points();
      group_map_points.insert(mps.begin(), mps.end());
    }
  }

  // project and match map points
  for (const auto & loop_kf: loop_key_frames) {
    const auto in_front_filter = [transform = loop_kf->world_to_camera()](const auto mp) {
        const auto img_p = transform * mp->position().homogeneous();
        return img_p.z() > 0 && img_p.x() >= 0 && img_p.x() <= 2000 && img_p.y() >= 0 &&
               img_p.y() <= 500;
      };

    size_t count = 0;
    for (const auto & mp: group_map_points | std::views::filter(in_front_filter)) {
      auto projection = project_map_point(*loop_kf, *mp);
      const auto features = loop_kf->get_features_within_radius(
        projection.x, projection.y, 8.0,
        true);

      if (features.size() > 0) {
        const auto mp_desc = mp->description();

        double min_dist = cv::norm(mp_desc, key_frame->get_point(0).second, cv::NORM_HAMMING);
        size_t min_idx = features[0];

        for (size_t idx: features) {
          double dist = cv::norm(mp_desc, key_frame->get_point(0).second, cv::NORM_HAMMING);
          if (dist < min_dist) {
            min_idx = idx;
            min_dist = dist;
          }
        }

        const auto kf_mp = loop_kf->corresponding_map_point(min_idx);
        if (kf_mp.has_value()) {
          if (group_map_points.find(kf_mp.value()) != group_map_points.end()) {
            // we do not want to remove map points from the group because
            // they are the originals and we are currenlty processing them
            // Just unlink the map point from the key frame
            loop_kf->remove_map_point(kf_mp.value());
            kf_mp.value()->remove_keyframe(loop_kf);
          } else {
            // map point is not in the group and matches in the group, needs
            // to be merged. Remove and link.
            map.remove_map_point(kf_mp.value());
          }
        }
        map.link_keyframe_to_map_point(loop_kf, min_idx, mp);
      }
    }

    // add edges in the covisibility graph
    map.update_covisibility(loop_kf);
  }

  // optimize


  std::cout << "LOOP CLOSURE DONE!" << std::endl;
  // clear key frame groups
  keyframe_groups.clear();
}

}
