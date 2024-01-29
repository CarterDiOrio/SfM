#ifndef INC_GUARD_MAP_HPP
#define INC_GUARD_MAP_HPP

#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>

#include "reconstruction/keyframe.hpp"
#include <pcl/point_types.h>

#include <iosfwd>

namespace sfm
{
/// @brief Models a 3D point in the map
class MapPoint
{
public:
  MapPoint(
    cv::Mat, Eigen::Vector3d position, size_t keyframe_id,
    Eigen::Vector3i color = {0, 0, 0});

  /// @brief Gets the position of the map point in the world frame
  /// @return the position of the map point in world frame
  inline Eigen::Vector3d position() const
  {
    return pos;
  }

  /// @brief Gets the ORB binary description of the map point
  /// @return The ORB binary description of the map point
  inline cv::Mat description() const
  {
    return desc;
  }

  /// @brief Gets the color of the map point
  /// @return the color
  inline Eigen::Vector3i get_color() const
  {
    return color;
  }

  /// @brief Adds another keyframe where this point is visible from
  /// @param k_id the id of the keyframe
  void add_keyframe(size_t k_id);

private:
  /// @brief The orb descriptor that best matches the
  cv::Mat desc;

  /// @brief The 3D position of the point in world space
  Eigen::Vector3d pos;

  /// @brief The RGB color of the world point
  Eigen::Vector3i color;

  /// @brief The ids of the keyframes this point is visible in
  std::vector<size_t> keyframe_ids;

  /// @brief the minimum distance the point can be observed at according to
  /// orb scale and invariant constraints
  double min{0.0};

  /// @brief The maximum distance the point can be observed at according
  /// to orb scale and invariant constraints
  double max{10.0};
};

/// @brief A map of keyframes and map points
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
  size_t add_keyframe(const KeyFrame & frame);

  /// @brief Adds map points to the map
  /// @param k_id the initial keyframe id to associate the map points with
  /// @param points the 3D locations of the points in the key frame's frame.
  /// @param descriptions the ORB descriptors of the map points
  void create_mappoints(
    size_t k_id,
    const std::vector<Eigen::Vector3d> points,
    const std::vector<Eigen::Vector3i> colors,
    const cv::Mat & descriptions,
    const std::vector<size_t> orig_idx);

  /// @brief Gets a keyframe with the given keyframe id
  /// @param k_id the id of the keyframe
  /// @return The keyframe
  inline KeyFrame & get_keyframe(size_t k_id)
  {
    return keyframes.at(k_id);
  }

  /// @brief Gets the map point with the given id
  /// @param m_id the id of the map point
  /// @return the map point
  inline MapPoint & get_mappoint(size_t m_id)
  {
    return mappoints.at(m_id);
  }

  friend std::ostream & operator<<(std::ostream & os, const Map & map);

  inline size_t size()
  {
    return mappoints.size();
  }

private:
  /// @brief all the key frames in the map
  std::vector<KeyFrame> keyframes;

  /// @brief all the map points in the map
  std::vector<MapPoint> mappoints;
};


/// @brief Writes the map to the ostream as X Y Z C1 C2 C3
/// @param os the ostrema
/// @param map the map to write
std::ostream & operator<<(std::ostream & os, const Map & map);
}

#endif
