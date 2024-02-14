#ifndef INC_GUARD_PLACE_RECOGNITION_HPP
#define INC_GUARD_PLACE_RECOGNITION_HPP

#include "DBoW2/DBoW2.h"
#include "reconstruction/keyframe.hpp"
#include <memory>
#include <vector>

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
  /// @return a vector of the best matches
  std::vector<std::shared_ptr<KeyFrame>> query(const KeyFrame & key_frame, size_t num_matches);

private:
  OrbVocabulary vocabulary;
  OrbDatabase database;
};
}

#endif
