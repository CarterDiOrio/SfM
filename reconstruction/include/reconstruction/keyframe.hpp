#ifndef INC_GUARD_KEYFRAME_HPP
#define INC_GUARD_KEYFRAME_HPP

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <optional>

#include "reconstruction/pinhole.hpp"

namespace sfm
{

class KeyFrame
{
public:
  /// @brief Instantiates a KeyFrame object
  /// @param K The camera calibration matrix
  /// @param T_kw The transform from the world frame to the camera frame
  /// @param keypoints The observed keypoints in the image
  /// @param descriptions The descriptions of each keypoint
  KeyFrame(
    PinholeModel K, Eigen::Matrix4d T_kw, std::vector<cv::KeyPoint> keypoints,
    cv::Mat descriptions,
    cv::Mat img);

  /// @brief Gets the 3x3 camera calibration matrix
  /// @return the camera calibration matrix
  inline PinholeModel camera_calibration()
  {
    return K;
  }

  /// @brief Gets the transform from the world to the camera
  /// @return The transform from the world to the camera
  inline Eigen::Matrix4d transform() const
  {
    return T_kw;
  }

  /// @brief Gets the number of keypoints in the keyframe
  /// @return The number of keypoints
  inline size_t num_keypoints()
  {
    return keypoints.size();
  }

  /// @brief Gets the keypoint and description
  /// @param i the id/idx of the keypoint
  /// @return a pair of {keypoint, description}
  inline std::pair<cv::KeyPoint, cv::Mat> operator[](int i)
  {
    return {
      keypoints[i],
      descriptors.row(i)
    };
  }

  /// @brief Matches keypoints with the other descriptions and returns the matches
  /// @param query_descriptiors the keypoint descriptions
  /// @param matcher the matcher to use
  /// @return a vector of the matches
  std::vector<cv::DMatch> match(
    const cv::Mat & query_descriptiors,
    const std::shared_ptr<cv::DescriptorMatcher> matcher) const;

  /// @brief Links a map point to a key point in the frame
  /// @param kp_idx the key point index
  /// @param map_point_id the map point id
  void link_map_point(size_t kp_idx, size_t map_point_id);

  /// @brief returns the map point id corresponding to the keypoint
  /// @param keypoint_idx the keypoint index
  /// @return the map point id
  std::optional<size_t> corresponding_map_point(size_t keypoint_idx) const;

  /// @brief Gets the keypoints from the keyframe
  /// @return the vector of key points
  inline const std::vector<cv::KeyPoint> & get_keypoints() const
  {
    return keypoints;
  }

  cv::Mat img;

private:
  /// @brief The camera intrinsics matrix
  PinholeModel K;

  /// @brief The homogenous transformation matrix from the world to the camera
  /// frame.
  Eigen::Matrix4d T_kw;

  /// @brief The features keypoints in the image
  std::vector<cv::KeyPoint> keypoints;

  /// @brief The descriptiors for each keypoint
  cv::Mat descriptors;

  // These following two dictionaries form a bidirectional relationship
  // between the key points and the map points if the key point corresponds
  // to a map point

  /// @brief key point idx to map point id
  std::unordered_map<size_t, size_t> kp_to_mp_index;

  /// @brief map point id to key point idx
  std::unordered_map<size_t, size_t> mp_to_kp_index;
};
}

#endif
