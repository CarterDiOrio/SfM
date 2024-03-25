/// \file
/// \brief This file contains the tracker class. Handles estimation of motion between
/// frames

#ifndef INC_GUARD_TRACKER_HPP
#define INC_GUARD_TRACKER_HPP

#include <vector>
#include <Eigen/Dense>
#include "reconstruction/frame.hpp"

namespace sfm
{

struct TrackerResult
{

  /// @brief The estimated pose of the camera
  Eigen::Matrix4d pose;
};

class Tracker
{

public:
  Tracker();

  TrackerResult track();
};

}

#endif
