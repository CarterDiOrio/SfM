#include "reconstruction/sfm.hpp"

#include <algorithm>
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

namespace sfm
{
Reconstruction::Reconstruction(ReconstructionOptions options)
: matcher{cv::BFMatcher::create(cv::NORM_HAMMING, true)},
  detector{cv::ORB::create(2000)},
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
  std::cout << "MAP SIZE: " << map.size() << std::endl;
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

  std::cout << "INITIAL MAP SIZE: " << map.size() << std::endl;
}

void Reconstruction::track_previous_frame(const cv::Mat & frame, const cv::Mat & depth)
{
  // 1. get orb features in frame
  const auto features = detect_features(frame, detector);
  std::cout << "SIZE BEFORE FILTER: " << features.keypoints.size() << "\n";
  const auto [keypoints, descriptors] = filter_features(features, depth, options.max_depth);
  std::cout << "SIZE AFTER FILTER: " << keypoints.size() << "\n";

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

  auto transformation = pnp(image_points, world_points);
  std::cout << transformation << std::endl;

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
  for (const auto & pair: kp_mp_pairs) {
    map.link_keyframe_to_map_point(shared_current, pair.first, pair.second);
  }

  std::cout << "N MATCHES: " << kp_mp_pairs.size() << " " <<
    shared_current->get_map_points().size() << "\n";

  track_local_map(shared_current);

  //6. add new map points
  const auto mps = shared_current->create_map_points();
  std::cout << "CREATING UNMATCHED: " << mps.size() << "\n";
  for (const auto & [idx, map_point]: mps) {
    map.add_map_point(map_point);
    map.link_keyframe_to_map_point(shared_current, idx, map_point);
  }

  std::cout << "total created: " << shared_current->get_map_points().size() << "\n";
  std::cout << "\n";

  previous_keyframe = current;
}

void Reconstruction::track_local_map(std::shared_ptr<KeyFrame> key_frame)
{
  const auto local_map = map.get_local_map(key_frame, 2);

  // map points in current keyframe
  auto current_map_points = key_frame->get_map_points();

  // get all the map points in the local map
  std::unordered_set<std::shared_ptr<MapPoint>> local_set;
  for (const auto local_kf: std::views::join(local_map)) {
    auto map_points = local_kf->get_map_points() |
      std::views::filter(in_container(current_map_points, true));
    local_set.insert(map_points.begin(), map_points.end());
  }

  // filter map points
  int count = 0;
  for (auto mp: local_set) {
    auto projection = project_map_point(*key_frame, *mp);
    if (projection.x >= 0 && projection.x <= 1280 && projection.y >= 0 && projection.y <= 720) {
      const auto features = key_frame->get_features_within_radius(projection.x, projection.y, 5.0);
      const auto mp_desc = mp->description();

      if (features.size() > 0) {

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
  }

  std::cout << "ADDITIONAL MATCHES: " << count << " " << key_frame->get_map_points().size() << "\n";
}

Eigen::Matrix4d Reconstruction::pnp(
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
  return transformation;
}


}
