#include "yolo.hpp"


namespace auto_aim
{
YOLO::YOLO(const std::string & config_path, bool debug) : detector_(config_path, debug) {}

std::list<Armor> YOLO::detect(const cv::Mat & img, int frame_count)
{
  return detector_.detect(img, frame_count);
}

}  // namespace auto_aim
