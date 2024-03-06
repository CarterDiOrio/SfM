#ifndef INC_GUARD_PLACE_RECOGNITION_HPP
#define INC_GUARD_PLACE_RECOGNITION_HPP

#include "DBoW2/DBoW2.h"
#include <DBoW2/BowVector.h>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "keyframe.fwd.hpp"
#include "reconstruction/keyframe.hpp"
#include "reconstruction/map.hpp"

namespace sfm
{

/// @brief A class that handles place recognition of KeyFrames using ORB features
class PlaceRecognition
{

public:
  PlaceRecognition(const std::string & vocabulary_path);

  /// @brief Query the database for the best matches
  /// @param key_frame the keyframe to query
  /// @param num_matches the number of matches to return
  /// @param threshold the similarty threshold. All matches with a BoW vector
  /// similarity less than this threshold will be discarded.
  /// @return a vector of the best matches
  std::vector<std::shared_ptr<KeyFrame>> query(
    const KeyFrame & key_frame,
    size_t num_matches,
    Map & map,
    double threshold = 0.0);

  /// @brief Add a keyframe to the database
  /// @param key_frame the keyframe to add
  void add(std::shared_ptr<KeyFrame> key_frame);

  /// @brief Converts descriptors to a bow vector
  /// @param descriptiors the descriptors to convert
  DBoW2::BowVector convert(const std::vector<cv::Mat> & descriptiors) const;

  /// @brief Scores two keyframes for the BoW similarity
  /// @param key_frame the keyframe to score
  /// @param other the other keyframe to score
  /// @return the score
  double score(const KeyFrame & key_frame, const KeyFrame & other) const;

  /// @brief Forbids a keyframe from being matched
  /// @param key_frame the keyframe to forbid
  void forbid(std::shared_ptr<KeyFrame> key_frame);

private:
  OrbDatabase database;

  std::unordered_map<size_t, std::shared_ptr<KeyFrame>> entry_key_frame_map;
  std::unordered_map<std::shared_ptr<KeyFrame>, size_t> key_frame_entry_map;

  struct QueryQroup
  {
    std::vector<std::shared_ptr<KeyFrame>> keyframes;
    std::unordered_set<std::shared_ptr<KeyFrame>> covisibility;
    double score{0.0};
    std::shared_ptr<KeyFrame> best_key_frame;
    double best_score{std::numeric_limits<double>::max()};
  };

};
}

#endif
