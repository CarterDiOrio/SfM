#ifndef INC_GUARD_MAP_HPP
#define INC_GUARD_MAP_HPP

#include <opencv2/features2d.hpp>

namespace sfm
{
/// @brief Models a 3D point in the map
class MapPoint
{
public:
  MapPoint(cv::Mat descriptor, cv::Mat3d position);

  /// @brief Gets the position of the map point in the world frame
  /// @return the position of the map point in world frame
  inline cv::Mat position()
  {
    return pos;
  }

  /// @brief Gets the ORB binary description of the map point
  /// @return The ORB binary description of the map point
  inline cv::Mat description()
  {
    return desc;
  }

private:
  /// @brief The orb descriptor that best matches the
  cv::Mat desc;

  /// @brief The 3D position of the point in world space
  cv::Mat3d pos;

  /// @brief the minimum distance the point can be observed at according to
  /// orb scale and invariant constraints
  double min{0.0};

  /// @brief The maximum distance the point can be observed at according
  /// to orb scale and invariant constraints
  double max{10.0};
};

class Map
{

};
}

#endif
