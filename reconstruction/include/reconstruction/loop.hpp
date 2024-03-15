#ifndef INC_GUARD_LOOP_CLOSING_HPP
#define INC_GUARD_LOOP_CLOSING_HPP

#include "reconstruction/keyframe.hpp"
#include "reconstruction/map.hpp"
#include "reconstruction/place_recognition.hpp"

namespace sfm
{
/// @brief Handles detecting loops and performing loop closure on the map
class LoopCloser
{
public:
  struct LoopCloserOptions
  {
    size_t covisibility_threshold;
    size_t inlier_threshold;
  };

  /// @brief Key Frame groups for loop closure
  struct KeyFrameGroup
  {
    /// @brief the key frames in the group
    std::unordered_set<KeyFramePtr> key_frames;

    std::unordered_set<KeyFramePtr> covisibility;

    /// @brief the map points in the group
    std::unordered_set<MapPointPtr> map_points;

    /// @brief the transformation from the world to the keyframe
    Eigen::Matrix4d T_wk;

    /// @brief whether or not the group was expanded
    bool expanded = false;

    /// @brief the number of times the group was expanded
    size_t expanded_count = 0;

    /// @brief marks the group for deletion
    bool should_delete = false;
  };

  LoopCloser(
    std::shared_ptr<PlaceRecognition> place_recognition,
    std::shared_ptr<Map> map,
    std::shared_ptr<cv::DescriptorMatcher> matcher,
    LoopCloserOptions options
  );

  /// @brief detects and closes loops
  /// @param key_frame the key frame to detect loops with
  void detect_loops(KeyFramePtr key_frame);

  /// @brief remove the key frame from any group
  /// @param key_frame the key frame to remove
  void remove_key_frame(KeyFramePtr key_frame);

private:
  const LoopCloserOptions options;
  const std::shared_ptr<PlaceRecognition> place_recognition;
  const std::shared_ptr<Map> map;
  std::vector<KeyFrameGroup> key_frame_groups;
  std::shared_ptr<cv::DescriptorMatcher> matcher;
  static constexpr size_t expansion_consistency_threshold = 3;


  /// @brief Detects key frames that are similar to other areas of the map
  /// @param key_frame the key frame to detect with.
  std::vector<KeyFramePtr> detect_candidate_keyframes(KeyFramePtr key_frame);

  /// @brief Creates or finds groups that the candidate key frames belong to
  /// @param key_frame the current key frame
  /// @param candidates the key frames that the current key frame is similar to.
  std::vector<KeyFrameGroup> find_groups(
    KeyFramePtr key_frame,
    const std::vector<KeyFramePtr> & candidates);

  /// @brief Checks each group for
  /// @param key_frame the current key frame
  /// @param groups key frame groups that have passed time consistency.
  /// @return Optionally a KeyFrameGroup if any have passed the check
  std::optional<KeyFrameGroup> group_geometric_check(
    KeyFramePtr key_frame,
    std::vector<KeyFrameGroup> & groups);

  /// @brief closes and optimizes the loop
  /// @param key_frame the current key frame
  /// @param group the current group
  void close_loop(KeyFramePtr key_frame, KeyFrameGroup & group);
};

}


#endif
