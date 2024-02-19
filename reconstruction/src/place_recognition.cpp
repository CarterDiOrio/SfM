#include "reconstruction/place_recognition.hpp"
#include <DBoW2/BowVector.h>
#include <DBoW2/DBoW2.h>
#include <DBoW2/QueryResults.h>
#include <algorithm>
#include <memory>
#include <opencv2/core/mat.hpp>
#include "reconstruction/keyframe.hpp"
#include <ranges>
#include <unordered_set>
#include <vector>

namespace sfm
{
PlaceRecognition::PlaceRecognition(const std::string & vocabulary_path)
{
  OrbVocabulary voc;
  voc.loadFromBinaryFile(vocabulary_path);
  database.setVocabulary(voc, false, 0);
}

void PlaceRecognition::add(std::shared_ptr<KeyFrame> key_frame)
{
  auto entry_id = database.add(key_frame->get_descriptors());
  entry_key_frame_map[entry_id] = key_frame;
}

std::vector<std::shared_ptr<KeyFrame>> PlaceRecognition::query(
  const KeyFrame & query_key_frame,
  size_t num_matches,
  Map & map,
  double threshold)
{
  DBoW2::QueryResults QueryResults;
  database.query(query_key_frame.get_descriptors(), QueryResults, num_matches);
  const auto query_kf = entry_key_frame_map.at(QueryResults[0].Id);

  // group matches by covisibility
  std::vector<QueryQroup> groups;
  for (const auto & result: QueryResults | std::views::drop(1)) {   // first element is always the query
    const auto key_frame = entry_key_frame_map.at(result.Id);
    auto cov = map.get_neighbors(key_frame) | std::views::filter(
      [&query_kf](const auto & kf) {
        return query_kf != kf;
      }
    );

    bool in_group = false;
    for (auto & group: groups) {
      if (group.covisibility.find(key_frame) != group.covisibility.end()) {
        group.keyframes.push_back(key_frame);
        group.covisibility.insert(cov.begin(), cov.end());
        if (result.Score > group.best_score) {
          group.best_score = result.Score;
          group.best_key_frame = key_frame;
        }
        group.score += result.Score;
        in_group = true;
        break;
      }
    }

    if (!in_group) {
      groups.push_back(
        {
          .keyframes = {key_frame},
          .covisibility = {cov.begin(), cov.end()},
          .score = result.Score,
          .best_key_frame = key_frame,
          .best_score = result.Score,
        });
    }
  }

  // get the max score
  const auto max_score = (*std::max_element(
      groups.begin(), groups.end(), [](const auto & a, const auto & b) {
        return a.score < b.score;
      })).score;

  // filter by max score and threshold
  groups.erase(
    std::remove_if(
      groups.begin(), groups.end(), [threshold, max_score](const auto & group) {
        return group.score < threshold && group.score < 0.75 * max_score;
      }),
    groups.end());

  std::unordered_set<std::shared_ptr<KeyFrame>> matches;
  for (const auto & group: groups) {
    matches.insert(group.best_key_frame);
  }

  return {matches.begin(), matches.end()};
}

DBoW2::BowVector PlaceRecognition::convert(const std::vector<cv::Mat> & descriptiors) const
{
  DBoW2::BowVector bow_vector;
  database.getVocabulary()->transform(descriptiors, bow_vector);
  return bow_vector;
}

double PlaceRecognition::score(const KeyFrame & key_frame, const KeyFrame & other) const
{
  return database.getVocabulary()->score(key_frame.get_bow_vector(), other.get_bow_vector());
}
}
