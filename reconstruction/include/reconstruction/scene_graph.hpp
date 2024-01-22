#ifndef INC_GUARD_SCENE_GRAPH_HPP
#define INC_GUARD_SCENE_GRAPH_HPP

#include "reconstruction/view.hpp"
#include "reconstruction/matching.hpp"

#include <opencv2/features2d.hpp>

namespace sfm
{
/// @brief Graph for reconstruction
class SceneGraph
{
public:
  /// @brief Initializes the scene graph with a seed match
  /// @param match the match to initialize the graph with
  SceneGraph(const Match & match);
};
}

#endif
