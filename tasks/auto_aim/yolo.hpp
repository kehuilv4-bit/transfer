#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <opencv2/opencv.hpp>

#include "detector.hpp"

namespace auto_aim
{
class YOLO
{
public:
  YOLO(const std::string & config_path, bool debug = true);

  std::list<Armor> detect(const cv::Mat & img, int frame_count = -1);

private:
  Detector detector_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_HPP
