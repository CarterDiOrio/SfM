#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

int main()
{
  std::cout << "Hello, World!" << std::endl;

  auto capture = cv::VideoCapture("output.mp4");

  while (1) {
    cv::Mat frame;
    capture >> frame;

    cv::imshow("Frame", frame);
    auto c = cv::waitKey(33);
    if (c == 27) {
      break;
    }
  }

  return 0;
}
