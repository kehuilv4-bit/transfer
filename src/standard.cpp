#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/dart_guide/guide_detector.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/config_reloader.hpp"

using namespace std::chrono;

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::CBoard cboard(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  dart_guide::GuideDetector guide_detector(config_path);

  // 配置热重载器
  tools::ConfigReloader reloader(config_path);
  reloader.add_callback([&](const YAML::Node & yaml) {
    aimer.reload(yaml);
    camera.reload(yaml);
  });

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  auto mode = io::Mode::idle;
  auto last_mode = io::Mode::idle;

  while (!exiter.exit()) {
    camera.read(img, t);
    q = cboard.imu_at(t - 1ms);
    mode = cboard.mode;

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", io::MODES[mode]);
      last_mode = mode;
    }

    // recorder.record(img, q, t);

    if (mode == io::Mode::dart) {
      // ===== 引导灯模式 =====
      auto guide_result = guide_detector.detect(img, q);

      io::Command command;
      if (guide_result.is_locked) {
        command.control = true;
        command.yaw = guide_result.yaw;
        command.pitch = guide_result.pitch;
        command.shoot = (std::abs(guide_result.pixel_error_x) < guide_detector.fire_threshold());

        tools::logger()->info(
          "[Guide] LOCKED yaw={:.2f}° pitch={:.2f}° fire={}",
          guide_result.yaw * 57.3, guide_result.pitch * 57.3, command.shoot);
      } else {
        command.control = false;
        command.shoot = false;
        command.yaw = 0;
        command.pitch = 0;
      }

      cboard.send(command);

    } else {
      // ===== 原有自瞄逻辑 =====
      solver.set_R_gimbal2world(q);

      Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

      auto armors = detector.detect(img);

      auto targets = tracker.track(armors, t);

      auto command = aimer.aim(targets, t, cboard.bullet_speed);

      cboard.send(command);
    }

    // 自动检测YAML文件变化并重载参数
    reloader.check();
  }

  return 0;
}