#include "reconstruction/recording.hpp"

#include <iostream>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>

namespace sfm::utils
{
RecordingReader::RecordingReader(const std::string & directory)
: directory{directory} {}

std::optional<std::pair<cv::Mat, cv::Mat>> RecordingReader::read_frames()
{
  std::filesystem::path dir(directory);
  std::filesystem::path cfile("color_" + std::to_string(frame_counter) + ".png");
  std::filesystem::path dfile("depth_" + std::to_string(frame_counter) + ".png");

  auto cpath = std::filesystem::canonical(dir / cfile);
  auto dpath = std::filesystem::canonical(dir / dfile);

  std::cout << cpath.string() << std::endl;
  std::cout << dpath.string() << std::endl;
  // if (!std::filesystem::exists(cfile) || !std::filesystem::exists(dfile)) {
  //   return {};
  // }

  auto color_image = cv::imread(cpath.string(), cv::IMREAD_COLOR);
  auto depth_image = cv::imread(dpath.string(), cv::IMREAD_ANYDEPTH);
  frame_counter += 1;
  return std::pair{
    color_image,
    depth_image
  };
}
}
