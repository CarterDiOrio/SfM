#ifndef INC_GUARD_KEYFRAME_HPP
#define INC_GUARD_KEYFRAME_HPP


#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <optional>
#include <memory>
#include <span>

#include "reconstruction/pinhole.hpp"
#include "reconstruction/keyframe.fwd.hpp"
#include "reconstruction/mappoint.fwd.hpp"

namespace sfm
{

class MapPoint;

class KeyFrame : public std::enable_shared_from_this<KeyFrame>
{
public:
  /// @brief Instantiates a KeyFrame object
  /// @param K The camera calibration matrix
  /// @param T_wk The transform from the world frame to the camera frame
  /// @param keypoints The observed keypoints in the image
  /// @param descriptions The descriptions of each keypoint
  KeyFrame(
    PinholeModel K,
    Eigen::Matrix4d T_wk,
    std::vector<cv::KeyPoint> keypoints,
    cv::Mat descriptions,
    cv::Mat img,
    cv::Mat depth);

  /// @brief Gets the 3x3 camera calibration matrix
  /// @return the camera calibration matrix
  PinholeModel camera_calibration() const;

  /// @brief Gets the transform from the cmaera to the world
  /// @return The transform from the camera to the world
  Eigen::Matrix4d transform() const;

  /// @brief Gets the transformation from the world to the camera frame
  /// @return the transformation
  Eigen::Matrix4d world_to_camera() const;

  /// @brief Sets the transformation from the world to the camera frame
  void set_world_to_camera(const Eigen::Matrix4d tf);

  /// @brief Gets the number of keypoints in the keyframe
  /// @return The number of keypoints
  size_t num_keypoints() const;

  /// @brief Gets the keypoint and description
  /// @param i the id/idx of the keypoint
  /// @return a pair of {keypoint, description}
  std::pair<cv::KeyPoint, cv::Mat> get_point(int i) const;

  /// @brief Matches keypoints with the other descriptions and returns the matches
  /// @param query_descriptiors the keypoint descriptions
  /// @param matcher the matcher to use
  /// @return a vector of the matches
  std::vector<cv::DMatch> match(
    const cv::Mat & query_descriptiors,
    const std::shared_ptr<cv::DescriptorMatcher> matcher) const;

  /// @brief Links a map point to a key point in the frame
  /// @param kp_idx the key point index
  /// @param map_point the map point
  void link_map_point(size_t kp_idx, std::shared_ptr<MapPoint> map_point);

  /// @brief returns the map point id corresponding to the keypoint
  /// @param keypoint_idx the keypoint index
  /// @return a shared pointer to the map point
  std::optional<std::shared_ptr<MapPoint>> corresponding_map_point(size_t keypoint_idx) const;

  /// @brief gets the observed location of a map point in image coordinates
  /// @param map_point the map point to get the observed location of
  /// @return a double of {x, y} pixel locations
  std::pair<double, double> get_observed_location(const std::shared_ptr<MapPoint> map_point) const;

  /// @brief Gets the keypoints from the keyframe
  /// @return the vector of key points
  const std::vector<cv::KeyPoint> & get_keypoints() const;

  /// @brief projects and creates map points from every key point that is not associated already
  /// @returns a vector of pairs of {corresponding keypoint idx, shared ptr to map point}
  std::vector<std::pair<size_t, std::shared_ptr<MapPoint>>> create_map_points();

  /// @brief gets the map points linked to the key frame
  /// @returns a vector of map points
  std::vector<std::shared_ptr<MapPoint>> get_map_points() const;

  /// @brief gets the indicies of the unmatched features within the radius of a point
  /// @param x the x location of the point
  /// @param y the y location of the point
  /// @param r the radius around the point
  /// @return the indicies of the features within the radius
  std::vector<size_t> get_features_within_radius(double x, double y, double r);

  const cv::Mat img;

private:
  /// @brief The camera intrinsics matrix
  const PinholeModel K;

  /// @brief The homogenous transformation matrix from the camera to the world
  Eigen::Matrix4d T_wk;

  /// @brief the homogenous transformation matrix from the world to the camera.
  Eigen::Matrix4d T_kw;

  /// @brief The features keypoints in the image
  const std::vector<cv::KeyPoint> keypoints;

  /// @brief The descriptiors for each keypoint
  const cv::Mat descriptors;

  /// @brief The depth image for the key frame
  const cv::Mat depth_img;

  // These following two dictionaries form a bidirectional relationship
  // between the key points and the map points if the key point corresponds
  // to a map point

  /// @brief key point idx to map point id
  std::unordered_map<size_t, std::shared_ptr<MapPoint>> kp_to_mp_index;

  /// @brief map point id to key point idx
  std::unordered_map<std::shared_ptr<MapPoint>, size_t> mp_to_kp_index;
};

/// @brief projects the map point into the image frame of the key frame
/// @param key_frame the key frame to project onto
/// @param map_point the map point to project
/// @return the 2D point in the image frame
cv::Point2d project_map_point(const KeyFrame & key_frame, const MapPoint & map_point);
}

#endif
