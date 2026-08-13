#ifndef AUTO_AIM__AT_NN_HPP
#define AUTO_AIM__AT_NN_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{
class ATNN : public YOLOBase
{
public:
  ATNN(const std::string & config_path, bool debug);

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  struct ClassInfo
  {
    ArmorType type;
    ArmorName name;
  };

  std::string device_, model_path_;
  bool debug_, use_roi_, use_traditional_;
  int input_width_, input_height_;

  const float default_score_threshold_ = 0.25F;
  const float default_nms_threshold_ = 0.3F;
  float score_threshold_, nms_threshold_;
  double min_confidence_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;

  cv::Rect roi_;
  cv::Point2f offset_, padding_;

  Detector detector_;

  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  ClassInfo decode_class(int class_id) const;
  Color decode_color(const cv::Mat & color_scores) const;
  bool check_name(const Armor & armor) const;
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AT_NN_HPP
