#ifndef INC_GUARD_SFM_HPP
#define INC_GUARD_SFM_HPP

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <optional>
#include <unordered_set>

#include "reconstruction/pinhole.hpp"
#include "reconstruction/map.hpp"
#include "reconstruction/keyframe.hpp"
#include "reconstruction/place_recognition.hpp"
#include "reconstruction/loop.hpp"

namespace sfm
{

/// @brief Options for the reconstruction
struct ReconstructionOptions
{
  /// @brief The camera model
  PinholeModel model;

  /// @brief The maximum valid depth for a feature (in meters)
  double max_depth;

  std::string place_recognition_voc;
};

/// @brief Key Frame groups for loop closure
struct KeyFrameGroup
{
  /// @brief the key frames in the group
  std::unordered_set<std::shared_ptr<KeyFrame>> key_frames;

  std::unordered_set<std::shared_ptr<KeyFrame>> covisibility;

  /// @brief the map points in the group
  std::unordered_set<std::shared_ptr<MapPoint>> map_points;

  /// @brief the transformation from the world to the keyframe
  Eigen::Matrix4d T_wk;

  /// @brief whether or not the group was expanded
  bool expanded = false;

  /// @brief the number of times the group was expanded
  size_t expanded_count = 0;
};

/// @brief Class that handles incremental 3D reconstruction
class Reconstruction
{
public:
  /// @brief Initializes a reconstruction
  /// @param model the camera model to use.
  Reconstruction(ReconstructionOptions options);

  /// @brief Adds a sequential frame to the reconstruction. It is expected
  /// that the frame comes after the previously added frame
  /// @param frame the frame to add
  /// @param depth the corresponding depth image of the frame
  void add_frame_ordered(const cv::Mat & frame, const cv::Mat & depth);

  /// @brief Gets a const reference to the internal map
  /// @return the map
  std::shared_ptr<Map> get_map()
  {
    return map;
  }

private:
  std::shared_ptr<cv::DescriptorMatcher> matcher;
  const std::shared_ptr<cv::ORB> detector;
  const PinholeModel model;
  const ReconstructionOptions options;

  /// @brief the number of times a group must be expanded in order to be consistent
  std::vector<KeyFrameGroup> keyframe_groups;

  std::weak_ptr<KeyFrame> previous_keyframe;

  std::shared_ptr<PlaceRecognition> place_recognition;
  std::shared_ptr<Map> map;
  std::shared_ptr<LoopCloser> loop_closer;

  /// @brief Initializes the 3D reconstruction
  /// @param frame the frame to initialize the reconstruction with
  /// @param depth the corresponding depth image
  void initialize_reconstruction(const cv::Mat & frame, const cv::Mat & depth);

  /// @brief Tracks the features from the previous keyframe and adds the keyframe
  /// @param frame the current frame
  /// @param depth the current depth image
  void track_previous_frame(const cv::Mat & frame, const cv::Mat & depth);

  /// @brief tracks and establishes a key frames link to its local map
  /// @param key_frame the key frame
  void track_local_map(std::shared_ptr<KeyFrame> key_frame);
};
}

#endif
