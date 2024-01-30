#include "reconstruction/sfm.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "reconstruction/mappoint.hpp"
#include "reconstruction/keyframe.hpp"

namespace sfm
{
Reconstruction::Reconstruction(PinholeModel model)
: matcher{cv::BFMatcher::create(cv::NORM_HAMMING)},
  detector{cv::ORB::create(10000)},
  model{model}
{}

void Reconstruction::add_frame_ordered(const cv::Mat & frame, const cv::Mat & depth)
{
  // If no prior frames, initialize map with first frame as the origin:
  if (map.is_empty()) {
    initialize_reconstruction(frame, depth);
    return;
  }

  // create keyframe process:
  // 1. get orb features in frame
  const auto & [keypoints, descriptions] = detect_features(frame, detector);
  const auto colors = extract_colors(frame, keypoints);

  //2. find matches to previous frame map points
  const auto previous_shared = previous_keyframe.lock();
  const auto matches = previous_shared->match(descriptions, matcher);

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
    descriptions,
    frame,
    depth
  );
  const auto shared_current = current.lock();

  //5. add matched map points to keyframe
  for (const auto & pair: kp_mp_pairs) {
    shared_current->link_map_point(pair.first, pair.second);
  }

  //6. add new map points
  const auto new_map_points = shared_current->create_map_points();
  for (const auto & [idx, map_point]: new_map_points) {
    map.add_map_point(map_point);
    shared_current->link_map_point(idx, map_point);
  }

  std::cout << "MAP SIZE: " << map.size() << std::endl;

  previous_keyframe = current;
}

void Reconstruction::initialize_reconstruction(const cv::Mat & frame, const cv::Mat & depth)
{
  // 1. Detect feautres in image
  const auto & [keypoints, descriptions] = detect_features(frame, detector);

  // 2. project points to 3d
  const auto points = deproject_keypoints(keypoints, depth, model);

  // 3. Create map points
  const auto initial = map.create_keyframe(
    model,
    Eigen::Matrix4d::Identity(),
    keypoints,
    descriptions,
    frame,
    depth
  );

  // 5. create and link new map points
  const auto shared_keyframe = initial.lock();
  const auto new_map_points = shared_keyframe->create_map_points();
  for (const auto & [idx, map_point]: new_map_points) {
    map.add_map_point(map_point);
    shared_keyframe->link_map_point(idx, map_point);
  }

  previous_keyframe = initial;

  std::cout << "INITIAL MAP SIZE: " << map.size() << std::endl;
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
  std::cout << "started pnp " << image_points.size() << " " << world_points.size() << std::endl;
  cv::solvePnPRansac(
    world_points_cv, image_points, k,
    cv::noArray(), rvec, tvec, false, 2000, 8.0, 0.99, inliers, cv::SOLVEPNP_SQPNP);

  int ni = std::count_if(
    inliers.begin(), inliers.end(),
    [](const int & i) {
      return i > 0;
    }
  );


  std::cout << "finished pnp " << ni << std::endl;

  // // convert screw to transformation matrix
  cv::Mat rotation_mat;
  cv::Rodrigues(rvec, rotation_mat);

  Eigen::Matrix3d rotation_eig;
  cv::cv2eigen(rotation_mat, rotation_eig);
  Eigen::Vector3d translation_eig{tvec.at<double>(0, 0), tvec.at<double>(1, 0),
    tvec.at<double>(2, 0)};

  Eigen::Matrix4d transformation;
  transformation.setIdentity();
  transformation.block<3, 3>(0, 0) = rotation_eig.transpose();
  transformation.block<3, 1>(0, 3) = -1 * rotation_eig.transpose() * translation_eig;
  return transformation;
}


}
