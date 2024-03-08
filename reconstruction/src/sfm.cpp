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

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <ranges>
#include <time.h>


namespace sfm
{
Reconstruction::Reconstruction(ReconstructionOptions options)
: matcher{cv::BFMatcher::create(cv::NORM_HAMMING, true)},
  detector{cv::ORB::create(2000
    )},
  model{options.model},
  options{options}
{
  map = std::make_shared<Map>();
  place_recognition = std::make_shared<PlaceRecognition>(options.place_recognition_voc);
  loop_closer = std::make_shared<LoopCloser>(
    place_recognition, map, matcher,
    LoopCloser::LoopCloserOptions{
      .covisibility_threshold = 30
    });
}


void Reconstruction::add_frame_ordered(const cv::Mat & frame, const cv::Mat & depth)
{
  // If no prior frames, initialize map with first frame as the origin:
  if (map->is_empty()) {
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
  const auto initial = map->create_keyframe(
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
    map->add_map_point(map_point);
  }

  // 6. set the bow vector for the place recognition
  shared_keyframe->set_bow_vector(place_recognition->convert(shared_keyframe->get_descriptors()));
  place_recognition->add(shared_keyframe);

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

  auto [transformation, inliers] = pnp(image_points, world_points, model);

  // filter for pnp inliers
  std::vector<std::pair<size_t, std::shared_ptr<MapPoint>>> filtered_kp_mp;
  for (const auto [idx, inlier_val]: std::views::enumerate(inliers)) {
    if (inlier_val > 0) {
      filtered_kp_mp.push_back(kp_mp_pairs[idx]);
    }
  }

  //4. create key frame
  const auto current = map->create_keyframe(
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
  shared_current->set_bow_vector(place_recognition->convert(shared_current->get_descriptors()));
  place_recognition->add(shared_current);

  //5. add matched map points to keyframe
  for (const auto & pair: filtered_kp_mp) { // only link the key points that were inliers in pnp
    map->link_keyframe_to_map_point(shared_current, pair.first, pair.second);
  }
  map->update_covisibility(shared_current); // update the covisibility graph

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

  const auto [local_transformation, local_inliers] = pnp(image_points, world_points, model);
  shared_current->set_world_to_camera(local_transformation.inverse());
  for (const auto [idx, mp]: std::views::enumerate(shared_current->get_map_points())) {
    // remove outliers
    if (local_inliers[idx] < 1) {
      map->unlink_kf_and_mp(shared_current, mp);
    }
  }
  map->update_covisibility(shared_current);

  // create new map points from unmatched points
  const auto mps = shared_current->create_map_points();
  for (const auto & [idx, map_point]: mps) {
    map->add_map_point(map_point);
  }
  map->update_covisibility(shared_current);

  // attempt to loop close
  loop_closer->detect_loops(shared_current);

  // check if local map needs to be pruned
  // for (const auto & kf: map->get_neighbors(shared_current)) {
  //   const auto is_redundant = map->check_keyframe_redundancy(kf);
  //   if (is_redundant) {
  //     map->remove_key_frame(kf);
  //     place_recognition->forbid(kf);
  //   }
  // }

  auto second = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch());

  previous_keyframe = current;
}

void Reconstruction::track_local_map(std::shared_ptr<KeyFrame> key_frame)
{
  const auto local_map = map->get_local_map(key_frame, 2, 15);

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
    map->get_key_frames().size() <<
    std::endl;

  // filter map points
  for (auto mp: local_set) {
    auto projection = project_map_point(*key_frame, *mp);
    const auto features = key_frame->get_features_within_radius(
      projection.x, projection.y, 2.0, false);

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
      map->link_keyframe_to_map_point(key_frame, min_idx, mp);
    }
  }

}
}
