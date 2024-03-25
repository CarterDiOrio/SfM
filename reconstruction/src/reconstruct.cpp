#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <boost/program_options.hpp>

#include "reconstruction/recording.hpp"
#include "reconstruction/pinhole.hpp"

namespace po = boost::program_options;

int main(int argc, char ** argv)
{
  po::options_description desc("Allowed options");

  desc.add_options()
  ("help", "help message")
  ("data", po::value<std::string>(), "path to the dataset folder")
  ("vocab", po::value<std::string>(), "path to the DBoW2 vocabulary to use")
  ;

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") || argc == 1) {
      std::cout << desc << std::endl;
      exit(1);
    }

  } catch (po::error & e) {
    std::cerr << "ERROR: " << e.what() << std::endl << desc << std::endl;
    return 1;
  }


  const std::string data_dir = vm["data"].as<std::string>();
  const std::string vocab_dir = vm["vocab"].as<std::string>();

  // // sfm::PinholeModel model{913.848, 913.602, 642.941, 371.196};
  // sfm::PinholeModel model{718.856, 718.856, 607.1928, 185.2157};
  // sfm::utils::RecordingReader reader{data_dir};

  // sfm::ReconstructionOptions options{
  //   model,
  //   50.0,
  //   vocab_dir,
  // };
  // sfm::Reconstruction reconstruction{options};

  // auto frames = reader.read_frames();
  // for (size_t i = 0; frames.has_value(); i++) {
  //   const auto & [c, d] = frames.value();
  //   if (i % 2 == 0) {
  //     reconstruction.add_frame_ordered(c, d);
  //     if (cv::waitKey(1) == 'q') {
  //       break;
  //     }
  //   }

  //   frames = reader.read_frames();
  // }

  // std::ofstream dense_file;
  // dense_file.open("./dense.txt");
  // auto map = reconstruction.get_map();
  // sfm::DenseReconstruction dense_reconstruction{map};
  // dense_reconstruction.reconstruct(dense_file);

  return 0;
}
