#include "mt_detector.hpp"

namespace auto_aim::multithread
{
MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: detector_(config_path, debug)
{}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  queue_.push({img.clone(), t});
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto [img, t] = queue_.pop();
  return {detector_.detect(img), t};
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto [img, t] = queue_.pop();
  return {img, detector_.detect(img), t};
}
}  // namespace auto_aim::multithread
