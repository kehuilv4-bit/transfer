#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>

namespace auto_aim
{
namespace multithread
{

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  auto model_path = yaml[yolo_name + "_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  centered_letterbox_ = yolo_name == "at_nn";

  auto model = core_.read_model(model_path);
  auto shape = model->input().get_partial_shape();
  input_width_ = 640;
  input_height_ = 640;
  if (shape.rank().is_static() && shape.size() == 4 && shape[2].is_static() &&
      shape[3].is_static()) {
    input_width_ = static_cast<int>(shape[3].get_length());
    input_height_ = static_cast<int>(shape[2].get_length());
  }
  if (yaml[yolo_name + "_input_width"]) {
    input_width_ = yaml[yolo_name + "_input_width"].as<int>();
  }
  if (yaml[yolo_name + "_input_height"]) {
    input_height_ = yaml[yolo_name + "_input_height"].as<int>();
  }

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
    // .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
    .scale(255.0);

  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));

  tools::logger()->info("[MultiThreadDetector] initialized !");
}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  auto x_scale = static_cast<double>(input_height_) / img.rows;
  auto y_scale = static_cast<double>(input_width_) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);
  auto pad_x = centered_letterbox_ ? (input_width_ - w) / 2 : 0;
  auto pad_y = centered_letterbox_ ? (input_height_ - h) / 2 : 0;

  // preproces
  auto padding_value = centered_letterbox_ ? 114 : 0;
  auto input =
    cv::Mat(input_height_, input_width_, CV_8UC3, cv::Scalar::all(padding_value));
  auto roi = cv::Rect(pad_x, pad_y, w, h);
  cv::resize(img, input(roi), {w, h});

  auto input_port = compiled_model_.input();
  auto infer_request = compiled_model_.create_infer_request();
  ov::Tensor input_tensor(
    ov::element::u8, {1, static_cast<size_t>(input_height_), static_cast<size_t>(input_width_), 3},
    input.data);

  infer_request.set_input_tensor(input_tensor);
  infer_request.start_async();
  queue_.push({img.clone(), t, std::move(infer_request)});
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto [img, t, infer_request] = queue_.pop();
  infer_request.wait();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  auto x_scale = static_cast<double>(input_height_) / img.rows;
  auto y_scale = static_cast<double>(input_width_) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {std::move(armors), t};
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto [img, t, infer_request] = queue_.pop();
  infer_request.wait();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  auto x_scale = static_cast<double>(input_height_) / img.rows;
  auto y_scale = static_cast<double>(input_width_) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {img, std::move(armors), t};
}

}  // namespace multithread

}  // namespace auto_aim
