#include "reconstruction/sfm.hpp"

namespace sfm
{
Reconstruction::Reconstruction(PinholeModel model)
: matcher{cv::NORM_HAMMING, true},
  detector{cv::ORB::create(2000)},
  model{model}
{}

void Reconstruction::add_frame_ordered(const cv::Mat & frame, const cv::Mat & depth)
{
  // If no prior frames, initialize map with first frame as the origin:
  if (map.is_empty()) {
    initialize_reconstruction(frame, depth);
  }

  // create keyframe process:
  // 1. get orb features in frame
  const auto & [keypoints, descriptions] = detect_features(frame, detector);

  //2. find matches to previous frame map points
  const auto previous_keyframe = map.get_keyframe(previous_k_id);
  const auto matches = previous_keyframe.match(descriptions, matcher);

  // 2.1 for each match get the map points
  std::vector<std::pair<size_t, size_t>> kp_mp_pairs;
  for (const auto & dmatch: matches) {
    const auto mp_id = previous_keyframe.corresponding_map_point(dmatch.trainIdx);
    if (mp_id.has_value()) {
      kp_mp_pairs.emplace_back(dmatch.queryIdx, mp_id);
    }
  }


  // 3. Compute PnP from map points to 2D locations in current frame
  // 4. Construct key frame from the current frame using transformation
  // from PnP
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
    Eigen::Matrix4d::Ones(),
    keypoints,
    descriptions
  };

  // 4. insert keyframe into map
  auto k_id = map.add_keyframe(initial);

  // 5. create map points
  map.create_mappoints(k_id, points, descriptions);

  previous_k_id = k_id;
}


}
