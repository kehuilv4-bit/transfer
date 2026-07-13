#include "trajectory.hpp"

#include <cmath>

namespace tools
{
namespace
{
constexpr double gravity = 9.8;
constexpr double drag_k = 0.47 * 1.169 * (2 * 3.14159 * 0.02125 * 0.02125) / 2 / 0.041;
constexpr int max_iterations = 100;
constexpr double tolerance = 1e-6;
constexpr double min_cos_theta = 1e-6;
}

Trajectory::Trajectory(const double v0, const double d, const double h)
{
  unsolvable = true;
  fly_time = 0;
  pitch = 0;

  if (v0 <= 0 || d < 0 || !std::isfinite(v0) || !std::isfinite(d) || !std::isfinite(h)) {
    return;
  }

  if (d == 0) {
    pitch = (h >= 0) ? M_PI / 2 : -M_PI / 2;
    fly_time = 0;
    unsolvable = false;
    return;
  }

  auto theta = std::atan(h / d);
  auto converged = false;

  for (int i = 0; i < max_iterations; ++i) {
    auto cos_theta = std::cos(theta);
    if (std::abs(cos_theta) < min_cos_theta) {
      return;
    }

    auto exp_term = std::exp(drag_k * d);
    auto t = (exp_term - 1) / (drag_k * v0 * cos_theta);
    if (!std::isfinite(t)) {
      return;
    }

    auto cos_theta_sq = cos_theta * cos_theta;
    auto delta_h = h - v0 * std::sin(theta) * t / cos_theta + 0.5 * gravity * t * t / cos_theta_sq;
    if (std::abs(delta_h) < tolerance) {
      converged = true;
      fly_time = t;
      break;
    }

    auto denominator =
      -(v0 * t) / cos_theta_sq + gravity * t * t / (v0 * v0) * std::sin(theta) / (cos_theta_sq * cos_theta);
    if (std::abs(denominator) < tolerance || !std::isfinite(denominator)) {
      return;
    }

    theta -= delta_h / denominator;
    if (!std::isfinite(theta)) {
      return;
    }
  }

  if (!converged) {
    return;
  }

  pitch = theta;
  unsolvable = false;
}

}  // namespace tools