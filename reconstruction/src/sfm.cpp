#include "reconstruction/sfm.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

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
  const auto previous_keyframe = map.get_keyframe(previous_k_id);
  const auto matches = previous_keyframe.match(descriptions, matcher);


  // 2.1 for each match get the map points
  std::vector<std::pair<size_t, size_t>> kp_mp_pairs;
  for (const auto & dmatch: matches) {
    const auto mp_id = previous_keyframe.corresponding_map_point(dmatch.trainIdx);
    if (mp_id.has_value()) {
      kp_mp_pairs.emplace_back(static_cast<size_t>(dmatch.queryIdx), mp_id.value());
    }
  }


  cv::Mat mimg;
  cv::drawMatches(
    frame, keypoints, previous_keyframe.img,
    previous_keyframe.get_keypoints(), matches, mimg);
  cv::imshow("matches", mimg);

  // 3. Compute PnP from map points to 2D locations in current frame
  std::vector<cv::Point2d> image_points;
  std::vector<Eigen::Vector3d> world_points;
  for (const auto & pair: kp_mp_pairs) {
    image_points.push_back(keypoints[pair.first].pt);
    world_points.push_back(map.get_mappoint(pair.second).position());
  }

  auto transformation = pnp(image_points, world_points);
  std::cout << transformation << std::endl;

  //4. create key frame
  KeyFrame current{
    model,
    transformation,
    keypoints,
    descriptions,
    frame
  };

  //5. add matched map points to keyframe
  for (const auto & pair: kp_mp_pairs) {
    current.link_map_point(pair.first, pair.second);
  }

  auto current_k_id = map.add_keyframe(current);

  //6. add new map points
  std::vector<Eigen::Vector3d> new_points;
  cv::Mat new_descriptors;
  std::vector<Eigen::Vector3i> new_colors;
  std::vector<size_t> orig_idx;
  const auto deprojected = deproject_keypoints(keypoints, depth, model);
  for (size_t i = 0; i < keypoints.size(); i++) {
    auto it = std::find_if(
      kp_mp_pairs.begin(), kp_mp_pairs.end(), [&i](const auto & pair) {
        return pair.first == i;
      });

    // check if it does not already corresponds to a map point
    if (it == kp_mp_pairs.end()) {
      new_points.push_back(deprojected[i]);
      new_descriptors.push_back(descriptions.row(i));
      new_colors.push_back(colors[i]);
      orig_idx.push_back(i);
    }
  }

  map.create_mappoints(current_k_id, new_points, new_colors, new_descriptors, orig_idx);

  std::cout << "MAP SIZE: " << map.size() << std::endl;

  previous_k_id = current_k_id;
}

void Reconstruction::initialize_reconstruction(const cv::Mat & frame, const cv::Mat & depth)
{
  // 1. Detect feautres in image
  const auto & [keypoints, descriptions] = detect_features(frame, detector);

  // 2. project points to 3d
  const auto points = deproject_keypoints(keypoints, depth, model);

  // 3. Create map points
  const KeyFrame initial{
    model,
    Eigen::Matrix4d::Identity(),
    keypoints,
    descriptions,
    frame
  };

  // 4. insert keyframe into map
  auto k_id = map.add_keyframe(initial);

  // 5. create map points
  const auto colors = extract_colors(frame, keypoints);

  map.create_mappoints(k_id, points, colors, descriptions, std::vector<size_t>{});

  previous_k_id = k_id;

  std::cout << "MAP SIZE: " << map.size() << std::endl;
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
