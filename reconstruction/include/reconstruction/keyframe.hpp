#ifndef INC_GUARD_KEYFRAME_HPP
#define INC_GUARD_KEYFRAME_HPP

#include <vector>
#include <opencv2/core.hpp>

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
    cv::Matx33d K, cv::Matx44d T_kw, std::vector<cv::KeyPoint> keypoints,
    std::vector<cv::Mat> descriptions);

  /// @brief Gets the 3x3 camera calibration matrix
  /// @return the camera calibration matrix
  inline cv::Matx33d camera_calibration()
  {
    return K;
  }

  /// @brief Gets the transform from the world to the camera
  /// @return The transform from the world to the camera
  inline cv::Matx44d transform()
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
      descriptions[i]
    };
  }

private:
  /// @brief The camera intrinsics matrix
  cv::Matx33d K;

  /// @brief The homogenous transformation matrix from the world to the camera
  /// frame.
  cv::Matx44d T_kw;

  /// @brief The features keypoints in the image
  std::vector<cv::KeyPoint> keypoints;

  /// @brief The descriptions of each keypoint
  std::vector<cv::Mat> descriptions;
};
}

#endif
