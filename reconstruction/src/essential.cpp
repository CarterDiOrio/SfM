#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/stereo.hpp>



std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> packMatches(
    const std::vector<cv::KeyPoint>& queryPoints,
    const std::vector<cv::KeyPoint>& trainingPoints,
    const std::vector<cv::DMatch>& matches
) {
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : matches)
    {
        pts1.push_back(queryPoints[match.queryIdx].pt);
        pts2.push_back(trainingPoints[match.trainIdx].pt);
    }

    return { pts1, pts2 };
}

void drawEpilines(cv::Mat& image, 
                  const std::vector<cv::Point2f>& pts, 
                  const std::vector<cv::Point3f> epilines
) {
    auto cols = image.cols;
    for (const auto& line: epilines) {
        auto y0 = static_cast<int>(-line.z / line.y);
        auto y1 = static_cast<int>((-line.z - line.x * cols) / line.y);
        cv::line(image, {0, y0}, {cols, y1}, {255, 0, 0}, 1);
    }

    for (const auto& pt: pts) {
        cv::circle(image, static_cast<cv::Point2d>(pt), 3, {0, 255, 0}, -1);
    }
}

cv::Mat createProjectionMatrix(
    const cv::Mat& K,
    const cv::Mat& R,
    const cv::Mat& T
) {
    cv::Mat P = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat Rt = R.t();
    cv::Mat t = -1 * Rt * T;
    
    Rt.copyTo(P(cv::Rect{0, 0, 3, 3}));
    t.copyTo(P(cv::Rect{3, 0, 1, 3}));
    P = K * P;
    return P;
}


int main() {
    // camera calibration (recorded)
    // cv::Mat K = (cv::Mat_<double>(3,3) << 913.848, 0.0, 642.941, 0.0, 913.602, 371.196, 0.0, 0.0, 1.0); 
    // cv::Mat K = (cv::Mat_<double>(3,3) << 3437.84, 0.0, 3127.19, 0.0, 3435.95, 2066.98, 0.0, 0.0, 1.0); 
    cv::Mat K = (cv::Mat_<double>(3,3) << 963.721, 0.0, 956.196, 0.0, 963.721, 544.624, 0.0, 0.0, 1.0); 
    // read in images
    // auto img_left = cv::imread("../experiments/capture_0.png");
    // auto img_right = cv::imread("../experiments/capture_1.png");
    // auto img_left = cv::imread("../courtyard/images/dslr_images_undistorted/DSC_0287.JPG");
    // auto img_right = cv::imread("../courtyard/images/dslr_images_undistorted/DSC_0289.JPG");
    auto img_left = cv::imread("../images/rsm2_Color.png");
    auto img_right = cv::imread("../images/rsm1_Color.png");

    // convert them to gray
    auto gray_left = cv::Mat();
    auto gray_right = cv::Mat();
    cv::cvtColor(img_left, gray_left, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_right, gray_right, cv::COLOR_BGR2GRAY);

    // create orb detector
    auto orbDetector = cv::ORB::create(1000000);

    // detect and compute features in each image
    std::vector<cv::KeyPoint> left_keypoints;
    cv::Mat left_descriptors;
    orbDetector->detectAndCompute(gray_left, cv::noArray(), left_keypoints, left_descriptors);
    
    std::vector<cv::KeyPoint> right_keypoints;
    cv::Mat right_descriptors;
    orbDetector->detectAndCompute(gray_right, cv::noArray(), right_keypoints, right_descriptors);

    // display feautres 
    auto features_left = cv::Mat();
    auto features_right = cv::Mat();

    // create feature matcher
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    // match keypoints
    std::vector<cv::DMatch> matches;
    // first argument is QUERY descriptors, second is TRAINING descriptors
    matcher->match(left_descriptors, right_descriptors, matches);

    // sort keypoints to find the best
    std::sort(matches.begin(), matches.end());
    std::vector<cv::DMatch> best_matches{matches.begin(), matches.end()};

    //display mathces
    cv::Mat matches_img;

    auto pair = packMatches(left_keypoints, right_keypoints, matches);
    const auto& [left_pts, right_pts] = pair;

    // find fundamental matrix
    cv::Mat mask;

    //f is given as [p2; 1]F[p1; 1] = 0
    cv::Mat F = cv::findFundamentalMat(left_pts, right_pts, cv::USAC_DEFAULT, 1.0, 0.99, 100000, mask);

    // filter the points to be the inliers
    mask = mask.t();
    std::vector<cv::Point2f> left_inliers, right_inliers;
    std::vector<cv::DMatch> match_inliers;
    for(int i = 0; i < left_pts.size(); i++) {
        if (mask.at<bool>(i) == 1) {
            left_inliers.push_back(left_pts[i]);
            right_inliers.push_back(right_pts[i]);
            match_inliers.push_back(matches[i]);
        }
    }

    cv::correctMatches(F, left_inliers, right_inliers, left_inliers, right_inliers);

    cv::drawMatches(img_left, left_keypoints, img_right, right_keypoints, match_inliers, matches_img);

    std::cout << left_inliers.size() << "\n";

    // find lines in right picture corresponding to left points
    std::vector<cv::Point3f> right_lines;
    cv::computeCorrespondEpilines(left_inliers, 1, F, right_lines);

    // find lines in left picture corresponding to right points
    std::vector<cv::Point3f> left_lines;
    cv::computeCorrespondEpilines(right_inliers, 2, F, left_lines);

    // display the epilines
    features_left  = img_left.clone();
    features_right = img_right.clone();
    drawEpilines(features_left, left_inliers, left_lines);
    drawEpilines(features_right, right_inliers, right_lines);

    // find essential matrix
    cv::Mat E = K.t() * F * K;
    cv::Mat R, t, poseMask;
    cv::recoverPose(E, right_pts, left_pts, K, R, t, poseMask);

    // create the projection matrix for each camera
    auto p1 = createProjectionMatrix(K, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));    
    auto p2 = createProjectionMatrix(K, R, t);

    // triangulate the points
    cv::Mat homogenous_points;
    std::vector<cv::Point2f> li, ri;
    for(int i = 0; i < left_pts.size(); i++) {
        if (mask.at<bool>(i) == 1) {
            li.push_back(left_pts[i]);
            ri.push_back(right_pts[i]);
        }
    }

    std::cout << li.size();

    // triangulate best matches
    const auto& [lp, rp] = packMatches(left_keypoints, right_keypoints, best_matches);

    cv::triangulatePoints(p1, p2, left_inliers, right_inliers, homogenous_points);

    cv::Mat world_points;
    cv::convertPointsFromHomogeneous(homogenous_points.t(), world_points);

    // get the colors 
    std::vector<cv::Vec3b> colors;
    for(const auto& pt: left_inliers) {
        colors.push_back(
            img_left.at<cv::Vec3b>(static_cast<cv::Point2d>(pt))
        );
    }

    // write the points to the file
    std::ofstream file;
    std::string filename = "./cloud.txt";
    file.open(filename.c_str());

    for(int i = 0; i < colors.size(); i++) {
        file << world_points.at<float>(i, 0) << ", ";
        file << world_points.at<float>(i, 1) << ", ";
        file << world_points.at<float>(i, 2) << ", ";
        file << (int)colors[i][2] << ", ";
        file << (int)colors[i][1] << ", ";
        file << (int)colors[i][0] << "\n";
    }

    file.close();
    
    double decimation = 1.0 / 8.0;
    // cv::resize(features_left, features_left, cv::Size(), decimation, decimation);
    // cv::resize(features_right, features_right, cv::Size(), decimation, decimation);
    // cv::resize(matches_img, matches_img, cv::Size(), decimation, decimation);
    cv::imshow("left", features_left);
    cv::imshow("right", features_right);
    cv::imshow("matches", matches_img);

    int k = cv::waitKey(0);
    return 1;
}