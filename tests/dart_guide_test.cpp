/**
 * @brief 飞镖引导灯检测器测试程序
 * 
 * 支持两种输入源：
 *   1. 使用相机实时采集（默认） —— 上场时使用
 *   2. 使用视频文件（传入 -v=xxx.mp4）—— 调试时使用
 * 
 * 使用方法：
 *   ./dart_guide_test configs/standard3.yaml              # 用相机
 *   ./dart_guide_test configs/standard3.yaml -v=demo.mp4  # 用视频
 */
#include <fmt/core.h>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/dart_guide/guide_detector.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                       | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | yaml配置文件路径 }"
  "{video v        |                       | 测试视频路径(不传则使用相机) }"
  "{show s         | true                  | 是否显示调试窗口 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto video_path = cli.get<std::string>("video");
  auto show = cli.get<bool>("show");

  tools::Exiter exiter;

  // 初始化引导灯检测器
  dart_guide::GuideDetector guide_detector(config_path);

  // 输入源选择
  std::unique_ptr<io::Camera> camera_ptr;
  cv::VideoCapture video_cap;
  bool use_video = !video_path.empty();

  if (use_video) {
    video_cap.open(video_path);
    if (!video_cap.isOpened()) {
      tools::logger()->error("[DartGuideTest] Cannot open video: {}", video_path);
      return -1;
    }
    tools::logger()->info("[DartGuideTest] Using video: {}", video_path);
  } else {
    camera_ptr = std::make_unique<io::Camera>(config_path);
    tools::logger()->info("[DartGuideTest] Using camera.");
  }

  // 可选：初始化串口通信
  std::unique_ptr<io::CBoard> cboard_ptr;
  try {
    cboard_ptr = std::make_unique<io::CBoard>(config_path);
    tools::logger()->info("[DartGuideTest] CBoard initialized for serial communication.");
  } catch (const std::exception & e) {
    tools::logger()->warn("[DartGuideTest] CBoard init failed: {}, running without serial.", e.what());
  }

  if (show) {
    cv::namedWindow("DartGuide", cv::WINDOW_NORMAL);
    cv::resizeWindow("DartGuide", 800, 600);
    cv::namedWindow("Mask", cv::WINDOW_NORMAL);
    cv::resizeWindow("Mask", 800, 600);
  }

  int frame_count = 0;

  while (!exiter.exit()) {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    if (use_video) {
      video_cap >> img;
      if (img.empty()) break;
      t = std::chrono::steady_clock::now();
    } else {
      camera_ptr->read(img, t);
      if (img.empty()) continue;
    }

    // 获取 IMU 四元数：有串口时用真实值，没有时用单位四元数（仅调试用）
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    if (cboard_ptr) {
      q = cboard_ptr->imu_at(t - 1ms);
    }

    auto result = guide_detector.detect(img, q);
    frame_count++;

    // 构造 Command（与自瞄发送格式一致）
    io::Command command;
    if (result.is_locked) {
      command.control = true;
      command.yaw = result.yaw;
      command.pitch = result.pitch;
      command.shoot = (std::abs(result.pixel_error_x) < guide_detector.fire_threshold());

      tools::logger()->info(
        "[DartGuideTest][{}] LOCKED center=({:.0f},{:.0f}) yaw={:.2f}° pitch={:.2f}° fire={}",
        frame_count, result.center.x, result.center.y,
        result.yaw * 57.3, result.pitch * 57.3, command.shoot);
    } else {
      command.control = false;
      command.shoot = false;
      command.yaw = 0;
      command.pitch = 0;

      tools::logger()->debug("[DartGuideTest][{}] SEARCHING...", frame_count);
    }

    // 如果有串口，发送数据
    if (cboard_ptr) {
      cboard_ptr->send(command);
    }

    // 调试显示
    if (show) {
      cv::Mat display = img.clone();

      if (result.is_locked) {
        cv::circle(display, result.center, 12, cv::Scalar(0, 0, 255), -1);
        // 画十字线
        cv::line(display, {(int)result.center.x - 20, (int)result.center.y},
                          {(int)result.center.x + 20, (int)result.center.y},
                          cv::Scalar(0, 255, 0), 2);
        cv::line(display, {(int)result.center.x, (int)result.center.y - 20},
                          {(int)result.center.x, (int)result.center.y + 20},
                          cv::Scalar(0, 255, 0), 2);

        std::string info = fmt::format(
          "LOCKED | yaw:{:.2f}° pitch:{:.2f}° fire:{}",
          result.yaw * 57.3, result.pitch * 57.3, command.shoot);
        cv::putText(display, info, cv::Point(30, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
      } else {
        cv::putText(display, "SEARCHING...", cv::Point(30, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
      }

      // 画画面中心十字
      int cx = img.cols / 2, cy = img.rows / 2;
      cv::line(display, {cx - 30, cy}, {cx + 30, cy}, cv::Scalar(255, 255, 255), 1);
      cv::line(display, {cx, cy - 30}, {cx, cy + 30}, cv::Scalar(255, 255, 255), 1);

      cv::imshow("DartGuide", display);
      cv::imshow("Mask", guide_detector.getMask());

      int key = cv::waitKey(use_video ? 30 : 1);
      if (key == 27 || key == 'q') break;
    } else {
      if (use_video) std::this_thread::sleep_for(30ms);
    }
  }

  tools::logger()->info("[DartGuideTest] Exited. Total frames: {}", frame_count);
  return 0;
}
