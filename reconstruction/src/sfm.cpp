#include "reconstruction/sfm.hpp"

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
#include "reconstruction/utils.hpp"

#include <unordered_set>
#include <vector>
#include <ranges>
#include <time.h>

namespace sfm
{
Reconstruction::Reconstruction(ReconstructionOptions options)
: matcher{cv::BFMatcher::create(cv::NORM_HAMMING, true)},
  detector{cv::ORB::create(3000)},
  model{options.model},
  options{options}
{}

void Reconstruction::add_frame_ordered(const cv::Mat & frame, const cv::Mat & depth)
{
  // If no prior frames, initialize map with first frame as the origin:
  if (map.is_empty()) {
    initialize_reconstruction(frame, depth);
    return;
  }
  track_previous_frame(frame, depth);
  // std::cout << "MAP SIZE: " << map.size() << std::endl;
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
    depth
  );

  // 5. create and link new map points
  const auto shared_keyframe = initial.lock();
  for (const auto & [idx, map_point]: shared_keyframe->create_map_points()) {
    map.add_map_point(map_point);
    map.link_keyframe_to_map_point(shared_keyframe, idx, map_point);
  }

  previous_keyframe = initial;

  // std::cout << "INITIAL MAP SIZE: " << map.size() << std::endl;
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
    depth
  );
  const auto shared_current = current.lock();

  //5. add matched map points to keyframe
  for (const auto & pair: filtered_kp_mp) { // only link the key points that were inliers in pnp
    map.link_keyframe_to_map_point(shared_current, pair.first, pair.second);
  }
  map.update_covisibility(shared_current); // update the covisibility graph

  track_local_map(shared_current);
  // map.local_bundle_adjustment(shared_current, model);

  const auto mps = shared_current->create_map_points();
  for (const auto & [idx, map_point]: mps) {
    map.add_map_point(map_point);
    map.link_keyframe_to_map_point(shared_current, idx, map_point);
  }
  map.update_covisibility(shared_current);

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

  // std::cout << "LOCAL MAP SIZE: " << local_set.size() << "\n";

  // filter map points
  int count = 0;
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
      count++;
    }
  }

  map.update_covisibility(key_frame);

  // std::cout << "ADDITIONAL MATCHES: " << count << " " << key_frame->get_map_points().size() << "\n";
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
    cv::noArray(), rvec, tvec, false, 2000, 8.0, 0.99, inliers, cv::SOLVEPNP_SQPNP);

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


}
