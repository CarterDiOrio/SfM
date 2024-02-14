#ifndef INC_GUARD_MAP_HPP
#define INC_GUARD_MAP_HPP

#include <memory>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>

#include "reconstruction/keyframe.hpp"
#include "reconstruction/mappoint.fwd.hpp"
#include "reconstruction/pinhole.hpp"
#include <pcl/point_types.h>

#include <iosfwd>
#include <unordered_map>

namespace sfm
{
constexpr unsigned int covisibility_minimum = 15;

using key_frame_set_t = std::vector<std::shared_ptr<KeyFrame>>;

/// @brief A map of keyframes and map points. The role of this class
/// is to handle the further book keeping needed to maintain a covisibility graph
class Map
{
public:
  Map();

  /// @brief Checks if the map is empty or not
  /// @return returns true if the map is empty
  bool is_empty();

  /// @brief Adds a KeyFrame and returns the id of the keyframe
  /// @param frame the key frame to add
  /// @return the id of the added key frame
  size_t add_keyframe(std::shared_ptr<KeyFrame> keyframe);

  /// @brief Creates a keyframe and returns a weak ptr to it.
  /// @param K the amera model
  /// @param T_wk the transformation from the camera frame to the world
  /// @param keypoints the keypoints in the frame
  /// @param descriptions the orb descriptors for the keypoints
  /// @param img the image
  /// @param depth the depth image
  /// @return A weak ptr to the key frame.
  std::weak_ptr<KeyFrame> create_keyframe(
    PinholeModel K,
    Eigen::Matrix4d T_kw,
    std::vector<cv::KeyPoint> keypoints,
    cv::Mat descriptions,
    cv::Mat img,
    cv::Mat depth);

  /// @brief Gets a keyframe with the given keyframe id
  /// @param k_id the id of the keyframe
  /// @return The keyframe
  inline std::weak_ptr<KeyFrame> get_keyframe(size_t k_id)
  {
    return keyframes.at(k_id);
  }

  /// @brief Gets the map point with the given id
  /// @param m_id the id of the map point
  /// @return the map point
  inline std::weak_ptr<MapPoint> get_mappoint(size_t m_id)
  {
    return mappoints.at(m_id);
  }

  /// @brief Adds a map point to the map
  /// @param map_point a shared pointer to the map point
  void add_map_point(std::shared_ptr<MapPoint> map_point);

  /// @brief links a map point to the key frame's key point
  /// @param key_frame the key frame to link to
  /// @param key_point_idx the idx of the key point within the key frame
  /// @param map_point the map point to link to
  void link_keyframe_to_map_point(
    std::shared_ptr<KeyFrame> key_frame,
    size_t key_point_idx,
    std::shared_ptr<MapPoint> map_point);

  /// @brief Adds the key frame to the covisibility graph
  /// @param key_frame the key frame to link
  void update_covisibility(std::shared_ptr<KeyFrame> key_frame);

  /// @brief gets the local map around a key frame
  /// @param key_frame the key frame to get the map around
  /// @param distance how many levels of neighbors to include
  /// @return a vector of key_frame_sets where each element represents
  /// a set of nodes that have 1 higher distance than the last
  std::vector<key_frame_set_t> get_local_map(
    std::shared_ptr<KeyFrame> key_frame,
    size_t distance);

  /// @brief performs bundle adjustment on the local map around a keyframe
  /// @param key_frame the key frame to perform bundle adjustment around
  void local_bundle_adjustment(
    std::shared_ptr<KeyFrame> key_frame, PinholeModel model);

  friend std::ostream & operator<<(std::ostream & os, const Map & map);

  inline size_t size()
  {
    return mappoints.size();
  }

private:
  /// @brief all the key frames in the map
  std::vector<std::shared_ptr<KeyFrame>> keyframes;

  /// @brief all the map points in the map
  std::vector<std::shared_ptr<MapPoint>> mappoints;

  /// @brief holds the covisiblity graph
  std::unordered_map<std::shared_ptr<KeyFrame>,
    std::vector<std::shared_ptr<KeyFrame>>> covisibility;

  /// @brief links two keyframes in the covisibiltiy graph
  /// @param key_frame_1 the first key frame
  /// @param key_frame_2 the second key frame
  void covisibility_insert(
    std::shared_ptr<KeyFrame> key_frame_1,
    std::shared_ptr<KeyFrame> key_frame_2);
};


/// @brief Writes the map to the ostream as X Y Z C1 C2 C3
/// @param os the ostrema
/// @param map the map to write
std::ostream & operator<<(std::ostream & os, const Map & map);
}

#endif
