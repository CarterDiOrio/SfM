#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>

#include "reconstruction/features.hpp"
#include "reconstruction/matching.hpp"
#include "reconstruction/verification.hpp"

int main()
{
  std::cout << "Hello, World!" << std::endl;
  cv::Mat K =
    (cv::Mat_<double>(3, 3) << 963.721, 0.0, 956.196, 0.0, 963.721, 544.624, 0.0, 0.0, 1.0);

  auto capture = cv::VideoCapture("output.mp4");

  cv::Mat frame;
  std::vector<std::shared_ptr<sfm::View>> views;

  size_t frame_counter = 0;
  size_t view_id = 0;
  while (capture.read(frame)) {
    if (frame_counter % 10 == 0) {
      views.push_back(
        std::make_shared<sfm::View>(frame.clone(), K, view_id++)
      );
    }
    frame_counter++;
  }

  sfm::FeatureDetector feature_detector{cv::ORB::create(10000)};
  sfm::Matcher matcher{cv::BFMatcher::create(cv::NORM_HAMMING2, true)};
  auto feature_views = feature_detector.detectFeaturesInImages(views);
  auto matches = sfm::naiveMatching(feature_views, matcher);

  std::vector<sfm::VerifiedMatch> verified_matches;
  for (const auto & match: matches) {
    const auto verified = sfm::verify(match, 15);
    if (verified.has_value()) {
      verified_matches.push_back(verified.value());
    }
  }

  std::cout << "Total Views: " << feature_views.size() << "\n";
  std::cout << "Total Matches: " << matches.size() << "\n";
  std::cout << "Verified Views: " << verified_matches.size() << "\n";

  for (const auto & vm: verified_matches) {
    const auto img1 = vm.match.view1.view->image.clone();
    const auto img2 = vm.match.view2.view->image.clone();
    cv::Mat out;
    cv::drawMatches(
      img1, vm.match.view1.features, img2, vm.match.view2.features, vm.match.matches, out,
      cv::Scalar::all(-1), 2, vm.inlier_mask);
    cv::resize(out, out, cv::Size(), 0.5, 0.5);
    cv::imshow("matches", out);
    cv::waitKey(0);
  }

  return 0;
}
