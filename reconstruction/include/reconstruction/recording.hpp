#ifndef INC_GUARD_RECORDING_CPP
#define INC_GUARD_RECORDING_CPP

#include <string>
#include <optional>
#include <opencv2/core.hpp>

namespace sfm::utils
{

/// @brief Allows for sequential fetching of frames from a recorded log
/// of color and depth images
class RecordingReader
{
public:
  RecordingReader(const std::string & directory, size_t start_frame = 0);

  /// @brief Optionally returns a pair {color image, depth image} if the files exist
  /// @return A pair of color and depth image
  std::optional<std::pair<cv::Mat, cv::Mat>> read_frames();

private:
  size_t frame_counter{0};
  std::string directory;
};
}

#endif
