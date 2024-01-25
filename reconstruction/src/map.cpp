#include "reconstruction/map.hpp"

namespace sfm
{
MapPoint::MapPoint(cv::Mat descriptor, cv::Mat3d position)
: desc{descriptor}, pos{position} {}
}
