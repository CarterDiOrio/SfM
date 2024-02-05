#include <algorithm>

/// @brief Creates a function that checks if an item is in the container
/// @return a function that takes in an element and takes returns true
/// if that element is in the container
auto in_container(const auto & container, bool invert = false)
{
  const auto filter_func = [&container, invert](const auto element) {
      return (std::find(
               std::begin(container),
               std::end(container),
               element) !=
             std::end(container)) != invert;
    };
  return filter_func;
}
