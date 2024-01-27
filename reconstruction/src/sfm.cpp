#include "reconstruction/sfm.hpp"

namespace sfm
{
Reconstruction::Reconstruction(PinholeModel model)
: matcher{cv::BFMatcher::create(cv::NORM_HAMMING, true)},
  detector{cv::ORB::create(2000)},
  model{model}
{}

void Reconstruction::add_frame_ordered(const cv::Mat & frame, const cv::Mat & depth)
{
  // If no prior frames, initialize map with first frame as the origin:
  if (map.is_empty()) {
    initialize_reconstruction(frame, depth);
  }

  //create keyframe process:
  //1. get orb features in frame
  //2. find matches to previous frame map points
  //3. Compute PnP from map points to 2D locations in current frame
  //4. Construct key frame from the current frame using transformation
  // from PnP
}

void Reconstruction::initialize_reconstruction(const cv::Mat & frame, const cv::Mat & depth)
{
  // 1. Detect feautres in image
  const auto & [keypoints, descriptions] = detect_features(frame, detector);

  // 2. project points to 3d
  const auto points = deproject_keypoints(keypoints, depth, model);

  // 3. Create map points


  // 4. insert keyframe and map points into map
}


}
