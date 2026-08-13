#include "at_nn.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>

#include "tools/logger.hpp"

namespace auto_aim
{
namespace
{
int read_optional_int(const YAML::Node & yaml, const std::string & key, int default_value)
{
  return yaml[key] ? yaml[key].as<int>() : default_value;
}

float read_optional_float(const YAML::Node & yaml, const std::string & key, float default_value)
{
  return yaml[key] ? yaml[key].as<float>() : default_value;
}

std::pair<int, int> static_nchw_size(const std::shared_ptr<ov::Model> & model)
{
  auto shape = model->input().get_partial_shape();
  if (
    shape.rank().is_static() && shape.size() == 4 && shape[2].is_static() && shape[3].is_static()) {
    return {static_cast<int>(shape[3].get_length()), static_cast<int>(shape[2].get_length())};
  }
  return {640, 640};
}
}  // namespace

ATNN::ATNN(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["at_nn_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  score_threshold_ = read_optional_float(yaml, "at_nn_score_threshold", default_score_threshold_);
  nms_threshold_ = read_optional_float(yaml, "at_nn_nms_threshold", default_nms_threshold_);
  use_traditional_ = yaml["use_traditional"] ? yaml["use_traditional"].as<bool>() : false;

  int x = yaml["roi"]["x"].as<int>();
  int y = yaml["roi"]["y"].as<int>();
  int width = yaml["roi"]["width"].as<int>();
  int height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  auto model = core_.read_model(model_path_);
  auto [model_width, model_height] = static_nchw_size(model);
  input_width_ = read_optional_int(yaml, "at_nn_input_width", model_width);
  input_height_ = read_optional_int(yaml, "at_nn_input_height", model_height);

  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, static_cast<int64_t>(input_height_), static_cast<int64_t>(input_width_), 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

  tools::logger()->info(
    "[ATNN] model={}, input={}x{}, device={}", model_path_, input_width_, input_height_, device_);
}

std::list<Armor> ATNN::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return {};
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) roi_.width = raw_img.cols;
    if (roi_.height == -1) roi_.height = raw_img.rows;
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  auto x_scale = static_cast<double>(input_height_) / bgr_img.rows;
  auto y_scale = static_cast<double>(input_width_) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);
  auto pad_x = (input_width_ - w) / 2;
  auto pad_y = (input_height_ - h) / 2;
  padding_ = cv::Point2f(pad_x, pad_y);

  auto input = cv::Mat(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::resize(bgr_img, input(cv::Rect(pad_x, pad_y, w, h)), {w, h});

  ov::Tensor input_tensor(
    ov::element::u8, {1, static_cast<size_t>(input_height_), static_cast<size_t>(input_width_), 3},
    input.data);

  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  if (output_shape.size() < 2) {
    tools::logger()->warn("[ATNN] unexpected output rank: {}", output_shape.size());
    return {};
  }

  cv::Mat output(
    static_cast<int>(output_shape[output_shape.size() - 2]),
    static_cast<int>(output_shape[output_shape.size() - 1]), CV_32F, output_tensor.data());

  return parse(scale, output, raw_img, frame_count);
}

std::list<Armor> ATNN::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  cv::Mat detections;
  if (output.cols == 18) {
    detections = output;
  } else if (output.rows == 18) {
    cv::transpose(output, detections);
  } else {
    tools::logger()->warn("[ATNN] unsupported output shape: {}x{}", output.rows, output.cols);
    return {};
  }

  std::vector<Armor> candidates;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int r = 0; r < detections.rows; r++) {
    auto row = detections.row(r);
    auto confidence = row.at<float>(0, 4);
    if (confidence < score_threshold_) continue;

    auto x1 = (row.at<float>(0, 0) - padding_.x) / scale;
    auto y1 = (row.at<float>(0, 1) - padding_.y) / scale;
    auto x2 = (row.at<float>(0, 2) - padding_.x) / scale;
    auto y2 = (row.at<float>(0, 3) - padding_.y) / scale;
    auto left = static_cast<int>(std::min(x1, x2));
    auto top = static_cast<int>(std::min(y1, y2));
    auto width = static_cast<int>(std::abs(x2 - x1));
    auto height = static_cast<int>(std::abs(y2 - y1));
    if (width <= 0 || height <= 0) continue;

    std::vector<cv::Point2f> model_keypoints;
    model_keypoints.reserve(4);
    for (int i = 0; i < 4; i++) {
      model_keypoints.emplace_back(
        (row.at<float>(0, 10 + i * 2) - padding_.x) / scale,
        (row.at<float>(0, 11 + i * 2) - padding_.y) / scale);
    }
    std::vector<cv::Point2f> keypoints = {
      model_keypoints[0], model_keypoints[3], model_keypoints[2], model_keypoints[1]};

    auto class_id = static_cast<int>(std::round(row.at<float>(0, 5)));
    auto class_info = decode_class(class_id);
    auto color = decode_color(row.colRange(6, 10));
    auto box = cv::Rect(left, top, width, height);

    Armor armor(0, confidence, box, keypoints);
    if (use_roi_) {
      armor = Armor(0, confidence, box, keypoints, offset_);
    }
    armor.class_id = class_id;
    armor.color = color;
    armor.name = class_info.name;
    armor.type = class_info.type;

    candidates.emplace_back(std::move(armor));
    confidences.emplace_back(confidence);
    boxes.emplace_back(box);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    auto armor = candidates[i];
    if (!check_name(armor)) continue;
    if (use_traditional_) detector_.detect(armor, bgr_img);
    armor.center_norm = get_center_norm(bgr_img, armor.center);
    armors.emplace_back(std::move(armor));
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);
  return armors;
}

ATNN::ClassInfo ATNN::decode_class(int class_id) const
{
  static const std::vector<ClassInfo> classes = {
    {small, sentry}, {small, two}, {small, three}, {small, four}, {small, five}, {small, outpost},
    {big, sentry},   {big, one},   {big, three},   {big, four},   {big, five},   {big, base}};

  if (class_id >= 0 && class_id < static_cast<int>(classes.size())) return classes[class_id];
  return {small, not_armor};
}

Color ATNN::decode_color(const cv::Mat & color_scores) const
{
  double score;
  cv::Point color_id;
  cv::minMaxLoc(color_scores, nullptr, &score, nullptr, &color_id);

  switch (color_id.x) {
    case 0:
      return blue;
    case 1:
      return red;
    case 2:
      return extinguish;
    case 3:
      return purple;
    default:
      return extinguish;
  }
}

bool ATNN::check_name(const Armor & armor) const
{
  return armor.name != ArmorName::not_armor && armor.confidence > min_confidence_;
}

cv::Point2f ATNN::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  return {center.x / bgr_img.cols, center.y / bgr_img.rows};
}

void ATNN::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  (void)frame_count;
  auto detection = img.clone();
  for (const auto & armor : armors) {
    cv::rectangle(detection, armor.box, cv::Scalar(0, 255, 0), 2);
    for (const auto & point : armor.points) {
      cv::circle(detection, point, 3, cv::Scalar(0, 0, 255), -1);
    }
    auto label = COLORS[armor.color] + " " + ARMOR_NAMES[armor.name];
    cv::putText(
      detection, label, armor.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255),
      1);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}

std::list<Armor> ATNN::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  auto w = static_cast<int>(bgr_img.cols * scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  padding_ = cv::Point2f((input_width_ - w) / 2, (input_height_ - h) / 2);
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim
