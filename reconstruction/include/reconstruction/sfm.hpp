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
  inline const Map & get_map() const
  {
    return map;
  }

private:
  const std::shared_ptr<cv::DescriptorMatcher> matcher;
  const std::shared_ptr<cv::ORB> detector;
  const PinholeModel model;
  const ReconstructionOptions options;
  PlaceRecognition place_recognition;

  /// @brief the number of times a group must be expanded in order to be consistent
  static constexpr size_t expansion_consistency_threshold = 6;
  std::vector<KeyFrameGroup> keyframe_groups;

  std::weak_ptr<KeyFrame> previous_keyframe;

  Map map;

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

  /// @brief Attempts to find a loop and close it
  /// @param key_frame the key frame to check for loops with
  void loop_closing(std::shared_ptr<KeyFrame> key_frame);

  /// @brief Detects loops in the map
  /// @param key_frame the key frame to check for loops with
  /// @return a vector candiate key frames
  std::vector<std::shared_ptr<KeyFrame>> loop_candidate_detection(
    std::shared_ptr<KeyFrame> key_frame);

  /// @brief refines loop candiates by checking for consecutive consistency
  /// @param key_frame the key frame to check for loops with
  /// @param candidates the candidate key frames
  std::vector<KeyFrameGroup> loop_candidate_refinment(
    const std::shared_ptr<KeyFrame> key_frame,
    const std::vector<std::shared_ptr<KeyFrame>> & candidates);

  /// @brief performs feature matching and geometric consistency checking
  std::optional<KeyFrameGroup> loop_candidate_geometric(
    std::shared_ptr<KeyFrame> key_frame,
    std::vector<KeyFrameGroup> & groups);

  /// @brief closes the loop and performs optimization
  /// @param key_frame the key frame to close the loop with
  /// @param group the group to close the loop with
  void loop_closure(std::shared_ptr<KeyFrame> key_frame, KeyFrameGroup & group);

  /// @brief Performs Perspective-n-Point between the image points and the world points
  /// @param image_points the 2D points in the image
  /// @param world_points the corresponding 3D world peoints
  /// @return the 4x4 transformation matrix of the camera to the world
  std::pair<Eigen::Matrix4d, std::vector<int>> pnp(
    const std::vector<cv::Point2d> & image_points,
    const std::vector<Eigen::Vector3d> & world_points);
};
}

#endif
