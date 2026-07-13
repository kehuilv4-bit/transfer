#ifndef DART_GUIDE__GUIDE_DETECTOR_HPP
#define DART_GUIDE__GUIDE_DETECTOR_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <string>

namespace dart_guide
{

// 引导灯检测结果
struct GuideResult
{
  bool is_locked;      // 是否稳定锁定目标
  cv::Point2f center;  // 目标中心点像素坐标
  float yaw;           // 世界坐标系下的 yaw 绝对角度（弧度），与自瞄发送格式一致
  float pitch;         // 世界坐标系下的 pitch 绝对角度（弧度），与自瞄发送格式一致
  float pixel_error_x; // 相对光心的水平像素偏差（调试用）
  float pixel_error_y; // 相对光心的垂直像素偏差（调试用）
};

class GuideDetector
{
public:
  explicit GuideDetector(const std::string & config_path);

  /**
   * @brief 对一帧图像进行引导灯识别
   * @param frame 输入的原始彩色图
   * @param q     当前帧 IMU 四元数（用于将方向转到世界坐标系）
   * @return GuideResult 包含锁定状态与世界坐标系下的绝对角度
   */
  GuideResult detect(const cv::Mat & frame, const Eigen::Quaterniond & q);

  // 调试用：获取中间图像
  cv::Mat getMask() const { return mask_; }
  cv::Mat getGray() const { return gray_; }

  // 获取开火像素阈值（供外部判断是否开火）
  double fire_threshold() const { return fire_pixel_threshold_; }

private:
  void preprocess(const cv::Mat & input);
  bool findTarget(const cv::Mat & input, cv::Point2f & best_center);

  // ---- 算法参数（可通过 yaml 配置） ----
  int gray_threshold_;     // 二值化阈值（引导灯为高亮绿色，灰度值高）
  double min_area_;        // 最小面积过滤
  int min_found_frame_;    // 连续检测帧数阈值（防抖）
  double min_circularity_; // 最小圆度
  double max_ratio_;       // 最大长宽比
  double fire_pixel_threshold_; // 开火死区（像素）

  // ---- 相机内参 ----
  double fx_, fy_, cx_, cy_;

  // ---- 相机-云台外参（R_camera2gimbal, t_camera2gimbal） ----
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;

  // ---- IMU-云台外参 ----
  Eigen::Matrix3d R_gimbal2imubody_;

  // ---- 内部状态 ----
  int found_count_ = 0;
  cv::Mat gray_, mask_;

  // ---- 像素→世界坐标系绝对角度解算 ----
  void pixel2world_angle(
    float px, float py, const Eigen::Matrix3d & R_gimbal2world,
    float & yaw, float & pitch) const;
};

}  // namespace dart_guide

#endif  // DART_GUIDE__GUIDE_DETECTOR_HPP
