#ifndef INC_GUARD_COORDINATE_FRAME_HPP
#define INC_GUARD_COORDINATE_FRAME_HPP

#include <Eigen/Dense>

template<typename T>
class CoordinateFrame : public T
{
public:
  CoordinateFrame(const T & t)
  : T{t}
  {
    to_frame = Eigen::Matrix4d::Identity();
    from_frame = Eigen::Matrix4d::Identity();
  }

  auto point_to_frame(const Eigen::Vector3d & point) const
  {
    return (to_frame * point.homogeneous()).hnormalized();
  }

  auto point_from_frame(const Eigen::Vector3d & point) const
  {
    return (from_frame * point.homogeneous()).hnormalized();
  }

private:
  Eigen::Matrix4d to_frame;
  Eigen::Matrix4d from_frame;
};

#endif
