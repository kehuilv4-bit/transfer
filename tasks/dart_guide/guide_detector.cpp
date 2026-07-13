#include "guide_detector.hpp"

#include <cmath>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace dart_guide
{

GuideDetector::GuideDetector(const std::string & config_path)
{
  auto yaml = tools::load(config_path);

  // 读取相机内参 camera_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  auto cm = tools::read<std::vector<double>>(yaml, "camera_matrix");
  fx_ = cm[0];
  fy_ = cm[4];
  cx_ = cm[2];
  cy_ = cm[5];
  tools::logger()->info("[GuideDetector] Camera intrinsics: fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}",
                        fx_, fy_, cx_, cy_);

  // 读取相机→云台外参 R_camera2gimbal (3x3行优先) 和 t_camera2gimbal (3x1)
  auto R_data = tools::read<std::vector<double>>(yaml, "R_camera2gimbal");
  auto t_data = tools::read<std::vector<double>>(yaml, "t_camera2gimbal");
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_data.data());
  t_camera2gimbal_ = Eigen::Vector3d(t_data.data());

  // 读取云台→IMU外参 R_gimbal2imubody (3x3行优先)
  auto Rg_data = tools::read<std::vector<double>>(yaml, "R_gimbal2imubody");
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(Rg_data.data());

  tools::logger()->info("[GuideDetector] R_camera2gimbal, t_camera2gimbal, R_gimbal2imubody loaded.");

  // 从 yaml 中读取引导灯参数，若不存在则使用默认值
  if (yaml["dart_guide"]) {
    auto node = yaml["dart_guide"];
    gray_threshold_ = node["gray_threshold"] ? node["gray_threshold"].as<int>() : 250;
    min_area_ = node["min_area"] ? node["min_area"].as<double>() : 50.0;
    min_found_frame_ = node["min_found_frame"] ? node["min_found_frame"].as<int>() : 3;
    min_circularity_ = node["min_circularity"] ? node["min_circularity"].as<double>() : 0.5;
    max_ratio_ = node["max_ratio"] ? node["max_ratio"].as<double>() : 1.8;
    fire_pixel_threshold_ = node["fire_pixel_threshold"] ? node["fire_pixel_threshold"].as<double>() : 5.0;
  } else {
    gray_threshold_ = 250;
    min_area_ = 50.0;       // 引导灯在画面里比较小，默认值调低
    min_found_frame_ = 3;
    min_circularity_ = 0.5;
    max_ratio_ = 1.8;
    fire_pixel_threshold_ = 5.0;
    tools::logger()->warn("[GuideDetector] dart_guide config not found, using defaults.");
  }

  tools::logger()->info(
    "[GuideDetector] Params: threshold={}, min_area={:.0f}, min_frame={}, "
    "min_circ={:.2f}, max_ratio={:.2f}, fire_px_thresh={:.1f}",
    gray_threshold_, min_area_, min_found_frame_, min_circularity_, max_ratio_, fire_pixel_threshold_);
}

void GuideDetector::preprocess(const cv::Mat & input)
{
  // 提取绿色通道（引导灯为高亮绿色，绿色通道能量最强）
  std::vector<cv::Mat> channels;
  cv::split(input, channels);
  // channels[0]=B, channels[1]=G, channels[2]=R
  gray_ = channels[1];  // 使用绿色通道作为灰度图

  // 高斯滤波平滑噪声
  cv::GaussianBlur(gray_, gray_, cv::Size(5, 5), 0);

  // 高阈值二值化提取高亮绿色核心
  cv::threshold(gray_, mask_, gray_threshold_, 255, cv::THRESH_BINARY);

  // 形态学开运算，去除细小噪点
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(mask_, mask_, cv::MORPH_OPEN, kernel);
}

bool GuideDetector::findTarget(const cv::Mat & input, cv::Point2f & best_center)
{
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  bool frame_found = false;
  double max_score = -1.0;

  for (const auto & contour : contours) {
    double area = cv::contourArea(contour);
    if (area < min_area_) continue;

    double perimeter = cv::arcLength(contour, true);
    double circularity = (4 * CV_PI * area) / (perimeter * perimeter);

    cv::RotatedRect min_rect = cv::minAreaRect(contour);
    float w = min_rect.size.width;
    float h = min_rect.size.height;
    float ratio = (w > h) ? (w / h) : (h / w);

    // 筛选：圆度足够高且长宽比不过大
    if (circularity > min_circularity_ && ratio < max_ratio_) {
      // 评分：圆度越高、长宽比越接近1、面积越大则分数越高
      double score = (circularity * 10.0) + (10.0 / ratio) + (area / 100.0);
      if (score > max_score) {
        max_score = score;
        best_center = min_rect.center;
        frame_found = true;
      }
    }
  }
  return frame_found;
}

void GuideDetector::pixel2world_angle(
  float px, float py, const Eigen::Matrix3d & R_gimbal2world,
  float & yaw, float & pitch) const
{
  // 步骤1: 像素坐标 → 相机坐标系归一化方向向量
  Eigen::Vector3d dir_in_camera(
    (px - cx_) / fx_,
    (py - cy_) / fy_,
    1.0
  );

  // 步骤2: R_camera2gimbal 旋转到云台坐标系（与 solver.cpp 一致）
  Eigen::Vector3d dir_in_gimbal = R_camera2gimbal_ * dir_in_camera;

  // 步骤3: R_gimbal2world 旋转到世界坐标系（与自瞄一致）
  //   R_gimbal2world = R_gimbal2imubody^T * R_imubody2imuabs * R_gimbal2imubody
  //   这里直接传入已经算好的 R_gimbal2world
  Eigen::Vector3d dir_in_world = R_gimbal2world * dir_in_gimbal;

  // 步骤4: 从世界坐标系方向向量中提取 yaw 和 pitch
  //   与自瞄 planner 中 atan2(xyz.y(), xyz.x()) 完全一致
  double x = dir_in_world[0];
  double y = dir_in_world[1];
  double z = dir_in_world[2];

  yaw = std::atan2(y, x);
  pitch = std::atan2(z, std::sqrt(x * x + y * y));
}

GuideResult GuideDetector::detect(const cv::Mat & frame, const Eigen::Quaterniond & q)
{
  preprocess(frame);

  cv::Point2f current_center(0, 0);
  bool frame_found = findTarget(frame, current_center);

  if (frame_found) {
    found_count_++;
  } else {
    found_count_ = 0;
  }

  // 计算 R_gimbal2world（与 solver.cpp set_R_gimbal2world 完全一致）
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  Eigen::Matrix3d R_gimbal2world =
    R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;

  GuideResult result;
  result.center = current_center;
  result.is_locked = (found_count_ >= min_found_frame_);
  result.pixel_error_x = current_center.x - (float)cx_;
  result.pixel_error_y = current_center.y - (float)cy_;

  if (frame_found) {
    pixel2world_angle(current_center.x, current_center.y, R_gimbal2world,
                      result.yaw, result.pitch);
  } else {
    result.yaw = 0;
    result.pitch = 0;
  }

  return result;
}

}  // namespace dart_guide
