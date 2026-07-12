#include "target.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
constexpr double OUTPOST_HEIGHT_STEP = 0.10;  // m
constexpr double OUTPOST_HEIGHT_PROCESS_NOISE = 1e-4;
constexpr double OUTPOST_DIRECTION_GATING_SPEED = 0.4;
constexpr double OUTPOST_DIRECTION_SWITCH_MARGIN = 0.05;
constexpr double OUTPOST_HEIGHT_MIN = -0.08;
constexpr double OUTPOST_HEIGHT_MAX = 0.28;
constexpr double OUTPOST_HEIGHT_MIN_GAP = 0.03;
constexpr double OUTPOST_HEIGHT_MAX_GAP = 0.17;

constexpr int STATE_X = 0;
constexpr int STATE_VX = 1;
constexpr int STATE_Y = 2;
constexpr int STATE_VY = 3;
constexpr int STATE_Z = 4;
constexpr int STATE_VZ = 5;
constexpr int STATE_YAW = 6;
constexpr int STATE_VYAW = 7;
constexpr int STATE_R = 8;
constexpr int STATE_L = 9;
constexpr int STATE_H = 10;
}  // namespace

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  priority(armor.priority),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  switch_count_(0),
  is_switch_(false),
  is_converged_(false),
  t_(t)
{
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  auto center_x = xyz[0] + radius * std::cos(ypr[0]);
  auto center_y = xyz[1] + radius * std::sin(ypr[0]);
  auto center_z = xyz[2];

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(kStateSize);
  x0 << center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, radius, 0, 0, 0, 0;

  if (is_outpost()) {
    // The outpost keeps three cyclic height slots aligned with armor ids 0/1/2.
    x0[outpost_height_index(0)] = 0.0;
    x0[outpost_height_index(1)] = 2.0 * OUTPOST_HEIGHT_STEP;
    x0[outpost_height_index(2)] = OUTPOST_HEIGHT_STEP;
  }

  if (P0_dig.rows() != kStateSize) {
    Eigen::VectorXd expanded = Eigen::VectorXd::Zero(kStateSize);
    expanded.head(std::min<int>(P0_dig.rows(), kStateSize)) =
      P0_dig.head(std::min<int>(P0_dig.rows(), kStateSize));
    P0_dig = expanded;
  }
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[STATE_YAW] = tools::limit_rad(c[STATE_YAW]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

Target::Target(double x, double vyaw, double radius, double h)
: name(ArmorName::not_armor),
  armor_type(ArmorType::small),
  priority(ArmorPriority::fifth),
  jumped(false),
  last_id(0),
  armor_num_(4),
  switch_count_(0),
  update_count_(0),
  is_switch_(false),
  is_converged_(false)
{
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(kStateSize);
  x0 << x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h, 0, 0;
  Eigen::VectorXd P0_dig = Eigen::VectorXd::Zero(kStateSize);
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[STATE_YAW] = tools::limit_rad(c[STATE_YAW]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(kStateSize, kStateSize);
  F(STATE_X, STATE_VX) = dt;
  F(STATE_Y, STATE_VY) = dt;
  F(STATE_Z, STATE_VZ) = dt;
  F(STATE_YAW, STATE_VYAW) = dt;

  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;
    v2 = 0.1;
  } else {
    v1 = 100;
    v2 = 400;
  }

  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kStateSize, kStateSize);
  Q(STATE_X, STATE_X) = a * v1;
  Q(STATE_X, STATE_VX) = b * v1;
  Q(STATE_VX, STATE_X) = b * v1;
  Q(STATE_VX, STATE_VX) = c * v1;
  Q(STATE_Y, STATE_Y) = a * v1;
  Q(STATE_Y, STATE_VY) = b * v1;
  Q(STATE_VY, STATE_Y) = b * v1;
  Q(STATE_VY, STATE_VY) = c * v1;
  Q(STATE_Z, STATE_Z) = a * v1;
  Q(STATE_Z, STATE_VZ) = b * v1;
  Q(STATE_VZ, STATE_Z) = b * v1;
  Q(STATE_VZ, STATE_VZ) = c * v1;
  Q(STATE_YAW, STATE_YAW) = a * v2;
  Q(STATE_YAW, STATE_VYAW) = b * v2;
  Q(STATE_VYAW, STATE_YAW) = b * v2;
  Q(STATE_VYAW, STATE_VYAW) = c * v2;

  if (is_outpost()) {
    for (int i = 0; i < armor_num_; i++) {
      Q(outpost_height_index(i), outpost_height_index(i)) = OUTPOST_HEIGHT_PROCESS_NOISE;
    }
  }

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[STATE_YAW] = tools::limit_rad(x_prior[STATE_YAW]);
    return x_prior;
  };

  if (convergened() && is_outpost() && std::abs(ekf_.x[STATE_VYAW]) > 2) {
    ekf_.x[STATE_VYAW] = ekf_.x[STATE_VYAW] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

void Target::update(const Armor & armor)
{
  if (is_outpost()) {
    std::array<double, 3> costs{};
    for (int i = 0; i < armor_num_; i++) {
      costs[i] = outpost_match_cost(armor, i);
    }

    int matched_id = 0;
    if (convergened() && std::abs(ekf_.x[STATE_VYAW]) > OUTPOST_DIRECTION_GATING_SPEED) {
      int expected_shift = outpost_expected_shift();
      if (
        expected_shift >= 0 &&
        costs[expected_shift] + OUTPOST_DIRECTION_SWITCH_MARGIN < costs[matched_id]) {
        matched_id = expected_shift;
      }
    } else {
      auto min_cost = std::numeric_limits<double>::max();
      for (int i = 0; i < armor_num_; i++) {
        if (costs[i] < min_cost) {
          min_cost = costs[i];
          matched_id = i;
        }
      }
    }

    if (matched_id != 0) {
      jumped = true;
      rotate_outpost_reference(matched_id);
    }

    is_switch_ = matched_id != 0;
    if (is_switch_) switch_count_++;

    last_id = matched_id;
    update_count_++;
    update_ypda(armor, 0);
    return;
  }

  int id = 0;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  for (int i = 0; i < 3; i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  if (id != 0) jumped = true;

  is_switch_ = id != last_id;
  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  update_ypda(armor, id);
}

void Target::update(const std::vector<Armor> & armors)
{
  if (armors.empty()) return;

  const auto armor_count = std::min(static_cast<int>(armors.size()), armor_num_);

  if (!is_outpost()) {
    for (int i = 0; i < armor_count; i++) update(armors[i]);
    return;
  }

  if (armor_count == 1) {
    update(armors.front());
    return;
  }

  struct MatchResult
  {
    double cost = std::numeric_limits<double>::max();
    int shift = 0;
    double z_base = 0.0;
    std::vector<int> ids;
  } best_match;

  for (int shift = 0; shift < armor_num_; shift++) {
    auto candidate = *this;
    candidate.rotate_outpost_reference(shift);

    std::vector<int> current_ids(armor_count, -1);
    std::vector<bool> used_ids(armor_num_, false);

    auto dfs = [&](auto && self, int armor_index) -> void {
      if (armor_index == armor_count) {
        double z_base = 0.0;
        for (int i = 0; i < armor_count; i++) {
          z_base +=
            armors[i].xyz_in_world[2] - candidate.outpost_height_offset(candidate.ekf_.x, current_ids[i]);
        }
        z_base /= armor_count;
        candidate.ekf_.x[STATE_Z] = z_base;

        double total_cost = 0.0;
        for (int i = 0; i < armor_count; i++) {
          total_cost += candidate.outpost_match_cost(armors[i], current_ids[i]);
        }

        if (total_cost < best_match.cost) {
          best_match = {total_cost, shift, z_base, current_ids};
        }
        return;
      }

      for (int id = 0; id < armor_num_; id++) {
        if (used_ids[id]) continue;

        used_ids[id] = true;
        current_ids[armor_index] = id;
        self(self, armor_index + 1);
        used_ids[id] = false;
      }
    };

    dfs(dfs, 0);
  }

  if (best_match.shift != 0) {
    jumped = true;
    rotate_outpost_reference(best_match.shift);
  }

  ekf_.x[STATE_Z] = best_match.z_base;
  is_switch_ = best_match.shift != 0;
  if (is_switch_) switch_count_++;

  if (!best_match.ids.empty()) last_id = best_match.ids.front();
  update_count_ += armor_count;

  for (int i = 0; i < armor_count; i++) {
    update_ypda(armors[i], best_match.ids[i]);
  }
}

bool Target::is_outpost() const { return name == ArmorName::outpost && armor_num_ == 3; }

void Target::rotate_outpost_reference(int id)
{
  if (!is_outpost() || id <= 0 || id >= armor_num_) return;

  Eigen::MatrixXd permutation = Eigen::MatrixXd::Identity(kStateSize, kStateSize);
  for (int row = 0; row < armor_num_; row++) {
    for (int col = 0; col < armor_num_; col++) {
      permutation(outpost_height_index(row), outpost_height_index(col)) = 0.0;
    }
  }
  for (int new_slot = 0; new_slot < armor_num_; new_slot++) {
    const int old_slot = (new_slot + id) % armor_num_;
    permutation(outpost_height_index(new_slot), outpost_height_index(old_slot)) = 1.0;
  }

  ekf_.x[STATE_YAW] = tools::limit_rad(ekf_.x[STATE_YAW] + id * 2 * CV_PI / armor_num_);
  ekf_.x = permutation * ekf_.x;
  rotate_outpost_covariance(id);
}

double Target::outpost_match_cost(const Armor & armor, int id) const
{
  const auto xyza = armor_xyza_list()[id];
  const auto pred_ypd = tools::xyz2ypd(xyza.head(3));

  const auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));
  const auto yaw_error = std::abs(tools::limit_rad(armor.ypd_in_world[0] - pred_ypd[0]));
  const auto pitch_error = std::abs(tools::limit_rad(armor.ypd_in_world[1] - pred_ypd[1]));
  const auto distance_error = std::abs(armor.ypd_in_world[2] - pred_ypd[2]) / 0.2;
  const auto z_error = std::abs(armor.xyz_in_world[2] - xyza[2]) / OUTPOST_HEIGHT_STEP;

  return angle_error + yaw_error + 0.5 * pitch_error + 0.25 * distance_error + 0.5 * z_error;
}

int Target::outpost_expected_shift() const
{
  if (ekf_.x[STATE_VYAW] > 0) return 2;
  if (ekf_.x[STATE_VYAW] < 0) return 1;
  return -1;
}

int Target::outpost_height_index(int id) const { return kOutpostHeightBaseIndex + id; }

double Target::outpost_height_offset(const Eigen::VectorXd & x, int id) const
{
  return x[outpost_height_index(id)];
}

void Target::rotate_outpost_covariance(int id)
{
  Eigen::MatrixXd permutation = Eigen::MatrixXd::Identity(kStateSize, kStateSize);
  for (int row = 0; row < armor_num_; row++) {
    for (int col = 0; col < armor_num_; col++) {
      permutation(outpost_height_index(row), outpost_height_index(col)) = 0.0;
    }
  }
  for (int new_slot = 0; new_slot < armor_num_; new_slot++) {
    const int old_slot = (new_slot + id) % armor_num_;
    permutation(outpost_height_index(new_slot), outpost_height_index(old_slot)) = 1.0;
  }
  ekf_.P = permutation * ekf_.P * permutation.transpose();
}

void Target::update_ypda(const Armor & armor, int id)
{
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};

  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[STATE_YAW] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

  ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[STATE_YAW] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return xyza_list;
}

bool Target::diverged() const
{
  auto r_ok = ekf_.x[STATE_R] > 0.05 && ekf_.x[STATE_R] < 0.5;
  if (is_outpost()) {
    std::array<double, 3> heights{
      ekf_.x[outpost_height_index(0)], ekf_.x[outpost_height_index(1)],
      ekf_.x[outpost_height_index(2)]};
    std::sort(heights.begin(), heights.end());

    const bool heights_in_range =
      heights.front() > OUTPOST_HEIGHT_MIN && heights.back() < OUTPOST_HEIGHT_MAX;
    const bool gaps_reasonable =
      heights[1] - heights[0] > OUTPOST_HEIGHT_MIN_GAP &&
      heights[2] - heights[1] > OUTPOST_HEIGHT_MIN_GAP &&
      heights[1] - heights[0] < OUTPOST_HEIGHT_MAX_GAP &&
      heights[2] - heights[1] < OUTPOST_HEIGHT_MAX_GAP;

    if (r_ok && heights_in_range && gaps_reasonable) return false;

    tools::logger()->debug(
      "[Target] outpost diverged r={:.3f}, heights=[{:.3f}, {:.3f}, {:.3f}]",
      ekf_.x[STATE_R], heights[0], heights[1], heights[2]);
    return true;
  }

  auto l_ok = ekf_.x[STATE_R] + ekf_.x[STATE_L] > 0.05 && ekf_.x[STATE_R] + ekf_.x[STATE_L] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug(
    "[Target] r={:.3f}, l={:.3f}", ekf_.x[STATE_R], ekf_.x[STATE_R] + ekf_.x[STATE_L]);
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  if (is_outpost()) {
    auto angle = tools::limit_rad(x[STATE_YAW] + id * 2 * CV_PI / armor_num_);
    auto armor_x = x[STATE_X] - x[STATE_R] * std::cos(angle);
    auto armor_y = x[STATE_Y] - x[STATE_R] * std::sin(angle);
    auto armor_z = x[STATE_Z] + outpost_height_offset(x, id);

    return {armor_x, armor_y, armor_z};
  }

  auto angle = tools::limit_rad(x[STATE_YAW] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[STATE_R] + x[STATE_L] : x[STATE_R];
  auto armor_x = x[STATE_X] - r * std::cos(angle);
  auto armor_y = x[STATE_Y] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[STATE_Z] + x[STATE_H] : x[STATE_Z];

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  if (is_outpost()) {
    auto angle = tools::limit_rad(x[STATE_YAW] + id * 2 * CV_PI / armor_num_);
    auto dx_da = x[STATE_R] * std::sin(angle);
    auto dy_da = -x[STATE_R] * std::cos(angle);
    auto dx_dr = -std::cos(angle);
    auto dy_dr = -std::sin(angle);

    Eigen::MatrixXd H_armor_xyza = Eigen::MatrixXd::Zero(4, kStateSize);
    H_armor_xyza(0, STATE_X) = 1.0;
    H_armor_xyza(0, STATE_YAW) = dx_da;
    H_armor_xyza(0, STATE_R) = dx_dr;
    H_armor_xyza(1, STATE_Y) = 1.0;
    H_armor_xyza(1, STATE_YAW) = dy_da;
    H_armor_xyza(1, STATE_R) = dy_dr;
    H_armor_xyza(2, STATE_Z) = 1.0;
    H_armor_xyza(2, outpost_height_index(id)) = 1.0;
    H_armor_xyza(3, STATE_YAW) = 1.0;

    Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
    Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
    Eigen::MatrixXd H_armor_ypda{
      {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
      {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
      {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
      {0, 0, 0, 1}
    };

    return H_armor_ypda * H_armor_xyza;
  }

  auto angle = tools::limit_rad(x[STATE_YAW] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[STATE_R] + x[STATE_L] : x[STATE_R];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;
  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  Eigen::MatrixXd H_armor_xyza = Eigen::MatrixXd::Zero(4, kStateSize);
  H_armor_xyza(0, STATE_X) = 1.0;
  H_armor_xyza(0, STATE_YAW) = dx_da;
  H_armor_xyza(0, STATE_R) = dx_dr;
  H_armor_xyza(0, STATE_L) = dx_dl;
  H_armor_xyza(1, STATE_Y) = 1.0;
  H_armor_xyza(1, STATE_YAW) = dy_da;
  H_armor_xyza(1, STATE_R) = dy_dr;
  H_armor_xyza(1, STATE_L) = dy_dl;
  H_armor_xyza(2, STATE_Z) = 1.0;
  H_armor_xyza(2, STATE_H) = dz_dh;
  H_armor_xyza(3, STATE_YAW) = 1.0;

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {0, 0, 0, 1}
  };

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
