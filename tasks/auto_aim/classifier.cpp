#include "classifier.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>

namespace auto_aim
{
Classifier::Classifier(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  auto model = yaml["classify_model"].as<std::string>();
  auto use_lenet = model.find("lenet") != std::string::npos;
  input_size_ = use_lenet ? 28 : 32;
  use_softmax_ = !use_lenet;

  net_ = cv::dnn::readNetFromONNX(model);
  auto ovmodel = core_.read_model(model);
  compiled_model_ = core_.compile_model(
    ovmodel, "AUTO", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

void Classifier::classify(Armor & armor)
{
  if (armor.pattern.empty()) {
    armor.name = ArmorName::not_armor;
    return;
  }

  cv::Mat gray;
  if (armor.pattern.channels() == 1)
    gray = armor.pattern;
  else
    cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

  cv::Mat input;
  if (input_size_ == 28) {
    cv::resize(gray, input, cv::Size(input_size_, input_size_));
  } else {
    input = cv::Mat(input_size_, input_size_, CV_8UC1, cv::Scalar(0));
    auto x_scale = static_cast<double>(input_size_) / gray.cols;
    auto y_scale = static_cast<double>(input_size_) / gray.rows;
    auto scale = std::min(x_scale, y_scale);
    auto h = static_cast<int>(gray.rows * scale);
    auto w = static_cast<int>(gray.cols * scale);

    if (h == 0 || w == 0) {
      armor.name = ArmorName::not_armor;
      return;
    }
    auto roi = cv::Rect(0, 0, w, h);
    cv::resize(gray, input(roi), {w, h});
  }

  auto blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());

  net_.setInput(blob);
  cv::Mat outputs = net_.forward();

  if (use_softmax_) {
    // softmax
    float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::exp(outputs - max, outputs);
    float sum = cv::sum(outputs)[0];
    outputs /= sum;
  }

  double confidence;
  cv::Point label_point;
  cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
  int label_id = label_point.x;

  armor.confidence = confidence;
  armor.name = label_to_name(label_id);
}

void Classifier::ovclassify(Armor & armor)
{
  if (armor.pattern.empty()) {
    armor.name = ArmorName::not_armor;
    return;
  }

  cv::Mat gray;
  cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

  // Resize image to 32x32
  auto input = cv::Mat(input_size_, input_size_, CV_8UC1, cv::Scalar(0));
  auto x_scale = static_cast<double>(input_size_) / gray.cols;
  auto y_scale = static_cast<double>(input_size_) / gray.rows;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(gray.rows * scale);
  auto w = static_cast<int>(gray.cols * scale);

  if (h == 0 || w == 0) {
    armor.name = ArmorName::not_armor;
    return;
  }

  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(gray, input(roi), {w, h});
  // Normalize the input image to [0, 1] range
  input.convertTo(input, CV_32F, 1.0 / 255.0);

  ov::Tensor input_tensor(
    ov::element::f32, {1, 1, static_cast<size_t>(input_size_), static_cast<size_t>(input_size_)},
    input.data);

  ov::InferRequest infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat outputs(1, static_cast<int>(output_shape[1]), CV_32F, output_tensor.data());

  if (use_softmax_) {
    // Softmax
    float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::exp(outputs - max, outputs);
    float sum = cv::sum(outputs)[0];
    outputs /= sum;
  }

  double confidence;
  cv::Point label_point;
  cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
  int label_id = label_point.x;

  armor.confidence = confidence;
  armor.name = label_to_name(label_id);
}

ArmorName Classifier::label_to_name(int label_id) const
{
  if (input_size_ == 28) {
    // RPS26 lenet labels: 1,2,3,4,5,outpost,guard,base,negative.
    static const std::vector<ArmorName> names{
      one, two, three, four, five, outpost, sentry, base, not_armor};
    if (label_id >= 0 && label_id < static_cast<int>(names.size())) return names[label_id];
    return not_armor;
  }

  if (label_id >= 0 && label_id <= static_cast<int>(not_armor)) {
    return static_cast<ArmorName>(label_id);
  }
  return not_armor;
}

}  // namespace auto_aim
