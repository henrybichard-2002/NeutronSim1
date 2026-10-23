#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "openmc/particle_data.h"
#include "openmc/random_lcg.h"
#include "openmc/source.h"

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double BOLTZMANN_MEV_PER_K = 8.617333262e-11;
constexpr double MAXWELL_CUTOFF_SIGMAS = 12.0;
constexpr double WALL_SAMPLE_OFFSET_CM = 1.0e-6;

// Bare nuclear rest masses in MeV/c^2, derived consistently from AME2020
// evaluated atomic masses. The Q-values use this same nuclear-mass convention.
constexpr double MASS_D = 1875.61294257;
constexpr double MASS_T = 2808.92113298;
constexpr double MASS_N = 939.56542052;
constexpr double MASS_P = 938.27208816;
constexpr double MASS_ALPHA = 3727.37940660;
constexpr double MASS_HE3 = 2808.39160743;
constexpr double Q_DT = MASS_D + MASS_T - MASS_N - MASS_ALPHA;
constexpr double Q_DD_N = 2.0 * MASS_D - MASS_N - MASS_HE3;
constexpr double Q_DD_P = 2.0 * MASS_D - MASS_P - MASS_T;
static_assert(Q_DT > 17.5892 && Q_DT < 17.5893);
static_assert(Q_DD_N > 3.2688 && Q_DD_N < 3.2689);
static_assert(Q_DD_P > 4.0326 && Q_DD_P < 4.0327);

struct Vec3 {
  double x {};
  double y {};
  double z {};
};

struct FourVector {
  double energy {};
  Vec3 momentum {};
};

struct ProfilePoint {
  double radius_fraction {};
  double density {};
};

struct Parameters {
  double anode_radius_cm {4.05};
  double cathode_radius_cm {0.73};
  double half_height_cm {17.5};
  double voltage_kev {90.0};
  double neutral_temperature_k {300.0};
  double tritium_fraction {0.5};
  double beam_target_weight {1.0};
  double beam_wall_weight {0.0};
  double beam_beam_weight {0.0};
  double source_strength {1.0};
  std::vector<ProfilePoint> neutral_profile {{0.0, 1.0}, {1.0, 1.0}};
  std::vector<Vec3> reactor_centers {{0.0, 0.0, 0.0}};
  std::string source_spectrum_file;
  std::vector<double> source_spectrum_bins_mev;
};

double dot(Vec3 a, Vec3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double magnitude(Vec3 value)
{
  return std::sqrt(dot(value, value));
}

Vec3 operator+(Vec3 a, Vec3 b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator*(Vec3 value, double scale)
{
  return {value.x * scale, value.y * scale, value.z * scale};
}

double random_unit(uint64_t* seed)
{
  return std::clamp(openmc::prn(seed), 0.0, std::nextafter(1.0, 0.0));
}

double sample_normal(uint64_t* seed)
{
  const double u1 = std::max(random_unit(seed), 1.0e-16);
  const double u2 = random_unit(seed);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI * u2);
}

Vec3 sample_isotropic(uint64_t* seed)
{
  const double cosine = 2.0 * random_unit(seed) - 1.0;
  const double sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
  const double phi = 2.0 * PI * random_unit(seed);
  return {sine * std::cos(phi), sine * std::sin(phi), cosine};
}

std::vector<double> parse_number_list(const std::string& text)
{
  std::vector<double> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    size_t consumed = 0;
    const double value = std::stod(item, &consumed);
    if (consumed != item.size() || !std::isfinite(value)) {
      throw std::invalid_argument("Malformed number in neutral_profile parameter");
    }
    values.push_back(value);
  }
  return values;
}

std::vector<ProfilePoint> parse_profile(const std::string& text)
{
  std::vector<ProfilePoint> points;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const size_t separator = item.find(':');
    if (separator == std::string::npos) {
      throw std::invalid_argument(
        "neutral_profile entries must be radius_fraction:density pairs");
    }
    const auto values = parse_number_list(
      item.substr(0, separator) + "," + item.substr(separator + 1));
    if (values.size() != 2 || values[0] < 0.0 || values[0] > 1.0 ||
        values[1] < 0.0) {
      throw std::invalid_argument("Invalid neutral_profile entry");
    }
    if (!points.empty() && values[0] <= points.back().radius_fraction) {
      throw std::invalid_argument(
        "neutral_profile radius fractions must be strictly increasing");
    }
    points.push_back({values[0], values[1]});
  }
  if (points.size() < 2 || points.front().radius_fraction != 0.0 ||
      points.back().radius_fraction != 1.0) {
    throw std::invalid_argument(
      "neutral_profile must cover radius fractions 0 through 1");
  }
  return points;
}

std::vector<Vec3> parse_reactor_centers(const std::string& text)
{
  std::vector<Vec3> centers;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, '|')) {
    const auto values = parse_number_list(item);
    if (values.size() != 2) {
      throw std::invalid_argument(
        "reactor_centers_cm entries must be x,y coordinate pairs");
    }
    centers.push_back({values[0], values[1], 0.0});
  }
  if (centers.empty()) {
    throw std::invalid_argument("reactor_centers_cm must not be empty");
  }
  return centers;
}

Parameters parse_parameters(const std::string& text)
{
  Parameters result;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ';')) {
    if (item.empty()) {
      continue;
    }
    const size_t separator = item.find('=');
    if (separator == std::string::npos) {
      throw std::invalid_argument(
        "Compiled source parameters must be semicolon-separated key=value pairs");
    }
    const std::string key = item.substr(0, separator);
    const std::string value = item.substr(separator + 1);
    size_t consumed = 0;
    auto read_number = [&]() {
      const double number = std::stod(value, &consumed);
      if (consumed != value.size() || !std::isfinite(number)) {
        throw std::invalid_argument("Invalid numeric parameter: " + key);
      }
      return number;
    };

    if (key == "source_spectrum_file") result.source_spectrum_file = value;
    else if (key == "source_spectrum_bins_mev") {
      result.source_spectrum_bins_mev = parse_number_list(value);
    }
    else if (key == "r_anode_cm") result.anode_radius_cm = read_number();
    else if (key == "r_cathode_cm") result.cathode_radius_cm = read_number();
    else if (key == "half_height_cm") result.half_height_cm = read_number();
    else if (key == "voltage_kev") result.voltage_kev = read_number();
    else if (key == "neutral_temperature_k") result.neutral_temperature_k = read_number();
    else if (key == "tritium_fraction") result.tritium_fraction = read_number();
    else if (key == "zone_beam_target") result.beam_target_weight = read_number();
    else if (key == "zone_beam_wall") result.beam_wall_weight = read_number();
    else if (key == "zone_beam_beam") result.beam_beam_weight = read_number();
    else if (key == "source_strength") result.source_strength = read_number();
    else if (key == "neutral_profile") result.neutral_profile = parse_profile(value);
    else if (key == "reactor_centers_cm") result.reactor_centers = parse_reactor_centers(value);
    else throw std::invalid_argument("Unknown compiled source parameter: " + key);
  }

  if (!(result.anode_radius_cm > result.cathode_radius_cm &&
        result.cathode_radius_cm > 0.0 && result.half_height_cm > 0.0 &&
        result.voltage_kev >= 0.0 && result.neutral_temperature_k >= 0.0 &&
        result.tritium_fraction >= 0.0 && result.tritium_fraction <= 1.0 &&
        result.beam_target_weight >= 0.0 && result.beam_wall_weight >= 0.0 &&
        result.beam_beam_weight >= 0.0 && result.source_strength > 0.0)) {
    throw std::invalid_argument("Compiled source parameters are outside valid ranges");
  }
  const double total_zone_weight =
    result.beam_target_weight + result.beam_wall_weight + result.beam_beam_weight;
  if (!(total_zone_weight > 0.0)) {
    throw std::invalid_argument("At least one interaction-zone weight must be positive");
  }
  if (result.beam_beam_weight > 0.0 &&
      !(result.tritium_fraction < 1.0)) {
    throw std::invalid_argument(
      "Beam-beam mode requires deuterium ions; a zero tritium fraction is valid");
  }
  if (result.source_spectrum_file.empty() !=
      result.source_spectrum_bins_mev.empty()) {
    throw std::invalid_argument(
      "Source spectrum recording requires both an output path and energy bins");
  }
  if (result.source_spectrum_bins_mev.size() == 1 ||
      std::adjacent_find(
        result.source_spectrum_bins_mev.begin(),
        result.source_spectrum_bins_mev.end(),
        std::greater_equal<double>()) != result.source_spectrum_bins_mev.end()) {
    throw std::invalid_argument(
      "Source spectrum energy bins must be strictly increasing");
  }
  return result;
}

// Bosch-Hale (Bosch & Hale, Nucl. Fusion 32 (1992), 611-631),
// with energy in keV and the resulting cross section in m^2.
enum class Reaction { DT_N, DD_P, DD_N };

struct Fit {
  double gamow;
  double lower_kev;
  double upper_kev;
  double switch_kev;
  double a[5];
  double b[4];
  double a_high[5];
  double b_high[4];
};

constexpr Fit DT_FIT {
  34.3827, 0.5, 4700.0, 550.0,
  {6.927e4, 7.454e8, 2.050e6, 5.2002e4, 0.0},
  {6.38e1, -9.95e-1, 6.981e-5, 1.728e-4},
  {-1.4714e6, 0.0, 0.0, 0.0, 0.0},
  {-8.4127e-3, 4.7983e-6, -1.0748e-9, 8.5184e-14}};
constexpr Fit DD_P_FIT {
  31.3970, 0.5, 5000.0, 5000.0,
  {5.5576e4, 2.1054e2, -3.2638e-2, 1.4987e-6, 1.8181e-10},
  {0.0, 0.0, 0.0, 0.0}, {}, {}};
constexpr Fit DD_N_FIT {
  31.3970, 0.5, 4900.0, 4900.0,
  {5.3701e4, 3.3027e2, -1.2706e-1, 2.9327e-5, -2.5151e-9},
  {0.0, 0.0, 0.0, 0.0}, {}, {}};

double bosch_hale_sigma(double energy_kev, Reaction reaction)
{
  const Fit* fit = nullptr;
  switch (reaction) {
  case Reaction::DT_N: fit = &DT_FIT; break;
  case Reaction::DD_P: fit = &DD_P_FIT; break;
  case Reaction::DD_N: fit = &DD_N_FIT; break;
  }
  if (!(energy_kev > fit->lower_kev) || !std::isfinite(energy_kev)) {
    return 0.0;
  }

  const bool high_energy_fit =
    reaction == Reaction::DT_N && energy_kev > fit->switch_kev;
  const double s_energy = std::min(energy_kev, fit->upper_kev);
  const double* a = high_energy_fit ? fit->a_high : fit->a;
  const double* b = high_energy_fit ? fit->b_high : fit->b;
  const double exponent = fit->gamow / std::sqrt(energy_kev);
  if (exponent > 700.0) {
    return 0.0;
  }
  const double numerator =
    a[0] + s_energy * (a[1] + s_energy * (a[2] +
      s_energy * (a[3] + s_energy * a[4])));
  const double denominator =
    1.0 + s_energy * (b[0] + s_energy * (b[1] +
      s_energy * (b[2] + s_energy * b[3])));
  if (!(denominator > 0.0)) {
    return 0.0;
  }
  const double s_factor = numerator / denominator;
  return s_factor / (energy_kev * std::exp(exponent)) * 1.0e-31;
}

double total_cross_section(double energy_kev, bool deuterium_target)
{
  if (deuterium_target) {
    return bosch_hale_sigma(energy_kev, Reaction::DD_N) +
           bosch_hale_sigma(energy_kev, Reaction::DD_P);
  }
  return bosch_hale_sigma(energy_kev, Reaction::DT_N);
}

// Keep channel-specific bounds so the larger DT cross section does not
// suppress acceptance of DD events in a pure-deuterium source.
double maximum_fusion_cross_section(bool deuterium_target)
{
  double maximum = 0.0;
  for (int i = 0; i <= 100000; ++i) {
    const double energy = 0.5 + 0.05 * i;
    maximum = std::max(
      maximum, total_cross_section(energy, deuterium_target));
  }
  return maximum * 1.01;
}

double profile_density(const Parameters& parameters, double radius_fraction)
{
  const auto& profile = parameters.neutral_profile;
  auto upper = std::upper_bound(profile.begin(), profile.end(), radius_fraction,
    [](double value, const ProfilePoint& point) {
      return value < point.radius_fraction;
    });
  if (upper == profile.begin()) return upper->density;
  if (upper == profile.end()) return profile.back().density;
  const auto& right = *upper;
  const auto& left = *(upper - 1);
  const double fraction =
    (radius_fraction - left.radius_fraction) /
    (right.radius_fraction - left.radius_fraction);
  return left.density + fraction * (right.density - left.density);
}

double maximum_profile_density(
  const Parameters& parameters, double minimum_radius_fraction)
{
  double maximum = profile_density(parameters, minimum_radius_fraction);
  for (const auto& point : parameters.neutral_profile) {
    if (point.radius_fraction >= minimum_radius_fraction) {
      maximum = std::max(maximum, point.density);
    }
  }
  return maximum;
}

double local_beam_energy_kev(const Parameters& parameters, double radius_cm)
{
  // This is the voltage magnitude expressed as ion kinetic energy per charge.
  // For positive ions moving inward, the signed cathode potential is negative.
  const double denominator =
    std::log(parameters.cathode_radius_cm / parameters.anode_radius_cm);
  return parameters.voltage_kev *
    std::log(radius_cm / parameters.anode_radius_cm) / denominator;
}

Vec3 ion_beta(double kinetic_kev, double mass_mev, Vec3 direction)
{
  const double kinetic_mev = kinetic_kev * 1.0e-3;
  const double total_energy = mass_mev + kinetic_mev;
  const double momentum =
    std::sqrt(kinetic_mev * (kinetic_mev + 2.0 * mass_mev));
  return direction * (momentum / total_energy);
}

FourVector four_vector(double kinetic_kev, double mass_mev, Vec3 direction)
{
  const double kinetic_mev = kinetic_kev * 1.0e-3;
  const double total_energy = mass_mev + kinetic_mev;
  const double momentum =
    std::sqrt(kinetic_mev * (kinetic_mev + 2.0 * mass_mev));
  return {total_energy, direction * momentum};
}

Vec3 sample_thermal_beta(uint64_t* seed, double mass_mev, double temperature_k)
{
  const double component_sigma =
    std::sqrt(BOLTZMANN_MEV_PER_K * temperature_k / mass_mev);
  if (component_sigma == 0.0) return {};
  const double max_speed = MAXWELL_CUTOFF_SIGMAS * component_sigma;
  for (;;) {
    Vec3 beta {
      component_sigma * sample_normal(seed),
      component_sigma * sample_normal(seed),
      component_sigma * sample_normal(seed)};
    if (magnitude(beta) <= max_speed) return beta;
  }
}

FourVector four_vector_from_beta(double mass_mev, Vec3 beta)
{
  const double beta_squared = dot(beta, beta);
  if (!(beta_squared < 1.0)) {
    throw std::runtime_error("Sampled target ion speed is not subluminal");
  }
  const double gamma = 1.0 / std::sqrt(1.0 - beta_squared);
  return {gamma * mass_mev, beta * (gamma * mass_mev)};
}

double invariant_mass_squared(FourVector first, FourVector second)
{
  const FourVector total {
    first.energy + second.energy, first.momentum + second.momentum};
  return total.energy * total.energy - dot(total.momentum, total.momentum);
}

double center_of_mass_energy_kev(
  FourVector first, FourVector second, double mass1, double mass2)
{
  const double s = invariant_mass_squared(first, second);
  return std::max(0.0, std::sqrt(std::max(s, 0.0)) - mass1 - mass2) * 1000.0;
}

double moller_velocity(FourVector first, FourVector second, double mass1, double mass2)
{
  const double scalar_product =
    first.energy * second.energy - dot(first.momentum, second.momentum);
  const double flux_squared =
    scalar_product * scalar_product - mass1 * mass1 * mass2 * mass2;
  const double denominator = first.energy * second.energy;
  return std::sqrt(std::max(0.0, flux_squared)) / denominator;
}

Vec3 sample_annular_position(
  uint64_t* seed, double inner_radius_cm, double outer_radius_cm,
  double half_height_cm)
{
  const double radius = std::sqrt(
    inner_radius_cm * inner_radius_cm +
    random_unit(seed) * (outer_radius_cm * outer_radius_cm -
                         inner_radius_cm * inner_radius_cm));
  const double theta = 2.0 * PI * random_unit(seed);
  return {radius * std::cos(theta), radius * std::sin(theta),
          half_height_cm * (2.0 * random_unit(seed) - 1.0)};
}

Vec3 sample_cylinder_position(uint64_t* seed, double radius_cm, double half_height_cm)
{
  const double radius = radius_cm * std::sqrt(random_unit(seed));
  const double theta = 2.0 * PI * random_unit(seed);
  return {radius * std::cos(theta), radius * std::sin(theta),
          half_height_cm * (2.0 * random_unit(seed) - 1.0)};
}

struct Event {
  Vec3 position;
  FourVector first;
  FourVector second;
  double mass1;
  double mass2;
  bool deuterium_deuterium;
};

Event sample_beam_target_event(
  uint64_t* seed, const Parameters& parameters, bool wall)
{
  Vec3 position;
  Vec3 beam_direction;
  double radius;
  if (wall) {
    const double theta = 2.0 * PI * random_unit(seed);
    radius = parameters.anode_radius_cm - WALL_SAMPLE_OFFSET_CM;
    position = {radius * std::cos(theta), radius * std::sin(theta),
                parameters.half_height_cm * (2.0 * random_unit(seed) - 1.0)};
    beam_direction = {-std::cos(theta), -std::sin(theta), 0.0};
  } else {
    position = sample_annular_position(
      seed, parameters.cathode_radius_cm, parameters.anode_radius_cm,
      parameters.half_height_cm);
    radius = std::hypot(position.x, position.y);
    beam_direction = {-position.x / radius, -position.y / radius, 0.0};
  }

  const bool deuterium_target =
    random_unit(seed) >= parameters.tritium_fraction;
  const double target_mass = deuterium_target ? MASS_D : MASS_T;
  const double kinetic_kev = wall
    ? parameters.voltage_kev : local_beam_energy_kev(parameters, radius);
  const auto first = four_vector(kinetic_kev, MASS_D, beam_direction);
  const auto target_beta = sample_thermal_beta(
    seed, target_mass, parameters.neutral_temperature_k);
  const auto second = four_vector_from_beta(target_mass, target_beta);
  return {position, first, second, MASS_D, target_mass, deuterium_target};
}

Event sample_beam_beam_event(uint64_t* seed, const Parameters& parameters)
{
  const double deuterium_fraction = 1.0 - parameters.tritium_fraction;
  const double dd_pair_weight = deuterium_fraction * deuterium_fraction;
  const double dt_pair_weight =
    2.0 * deuterium_fraction * parameters.tritium_fraction;
  const bool dd_pair =
    random_unit(seed) < dd_pair_weight / (dd_pair_weight + dt_pair_weight);

  const double mass1 = dd_pair ? MASS_D :
    (random_unit(seed) < 0.5 ? MASS_D : MASS_T);
  const double mass2 = dd_pair ? MASS_D :
    (mass1 == MASS_D ? MASS_T : MASS_D);
  const Vec3 position =
    sample_cylinder_position(seed, parameters.cathode_radius_cm,
                             parameters.half_height_cm);
  const auto first = four_vector(
    parameters.voltage_kev, mass1, sample_isotropic(seed));
  const auto second = four_vector(
    parameters.voltage_kev, mass2, sample_isotropic(seed));
  return {position, first, second, mass1, mass2, dd_pair};
}

double event_acceptance(
  const Event& event, const Parameters& parameters, double sigma_max,
  double profile_max, bool beam_target_mode)
{
  const double e_cm = center_of_mass_energy_kev(
    event.first, event.second, event.mass1, event.mass2);
  const double sigma =
    total_cross_section(e_cm, event.deuterium_deuterium);
  if (!(sigma > 0.0)) return 0.0;
  const double velocity =
    moller_velocity(event.first, event.second, event.mass1, event.mass2);
  double density_factor = 1.0;
  if (beam_target_mode && profile_max > 0.0) {
    const double radius_fraction =
      std::hypot(event.position.x, event.position.y) /
      parameters.anode_radius_cm;
    density_factor =
      profile_density(parameters, radius_fraction) / profile_max;
  }
  return std::clamp(
    density_factor * (sigma / sigma_max) * (velocity / 2.0), 0.0, 1.0);
}

bool is_neutron_branch(uint64_t* seed, const Event& event)
{
  if (!event.deuterium_deuterium) return true;
  const double e_cm = center_of_mass_energy_kev(
    event.first, event.second, event.mass1, event.mass2);
  const double sigma_n = bosch_hale_sigma(e_cm, Reaction::DD_N);
  const double sigma_p = bosch_hale_sigma(e_cm, Reaction::DD_P);
  const double sigma_total = sigma_n + sigma_p;
  return sigma_total > 0.0 && random_unit(seed) < sigma_n / sigma_total;
}

openmc::SourceSite make_neutron_site(uint64_t* seed, const Event& event)
{
  const bool dt = !event.deuterium_deuterium;
  const double residual_mass = dt ? MASS_ALPHA : MASS_HE3;
  const double total_energy = event.first.energy + event.second.energy;
  const Vec3 total_momentum =
    event.first.momentum + event.second.momentum;
  const double s = total_energy * total_energy -
    dot(total_momentum, total_momentum);
  const double root_s = std::sqrt(std::max(s, 0.0));
  const double lambda =
    (s - (MASS_N + residual_mass) * (MASS_N + residual_mass)) *
    (s - (MASS_N - residual_mass) * (MASS_N - residual_mass));
  if (!(lambda >= 0.0 && root_s > 0.0)) {
    throw std::runtime_error("Fusion products are kinematically inaccessible");
  }

  const double momentum_cm = std::sqrt(lambda) / (2.0 * root_s);
  const double neutron_energy_cm =
    std::sqrt(momentum_cm * momentum_cm + MASS_N * MASS_N);
  const Vec3 direction_cm = sample_isotropic(seed);
  const Vec3 momentum_cm_vector = direction_cm * momentum_cm;
  const Vec3 beta_cm = total_momentum * (1.0 / total_energy);
  const double beta_squared = dot(beta_cm, beta_cm);
  const double gamma =
    1.0 / std::sqrt(std::max(1.0e-16, 1.0 - beta_squared));
  const double beta_dot_momentum = dot(beta_cm, momentum_cm_vector);
  const double boost_scale =
    beta_squared > 0.0
      ? ((gamma - 1.0) * beta_dot_momentum / beta_squared +
         gamma * neutron_energy_cm)
      : 0.0;
  const Vec3 momentum_lab =
    momentum_cm_vector + beta_cm * boost_scale;
  const double energy_lab =
    gamma * (neutron_energy_cm + beta_dot_momentum);
  const double kinetic_lab = energy_lab - MASS_N;
  const double momentum_norm = magnitude(momentum_lab);
  if (!(kinetic_lab > 0.0 && momentum_norm > 0.0)) {
    throw std::runtime_error("Computed neutron state is not physical");
  }

  openmc::SourceSite site;
  site.r = {event.position.x, event.position.y, event.position.z};
  site.u = {momentum_lab.x / momentum_norm, momentum_lab.y / momentum_norm,
            momentum_lab.z / momentum_norm};
  site.E = kinetic_lab * 1.0e6;
  site.wgt = 1.0;
  site.delayed_group = 0;
  site.particle = openmc::ParticleType(openmc::PDG_NEUTRON);
  return site;
}

} // namespace

class IecFusionSource final : public openmc::Source {
public:
  explicit IecFusionSource(const std::string& parameter_text)
    : parameters_ {parse_parameters(parameter_text)},
      sigma_max_dt_ {maximum_fusion_cross_section(false)},
      sigma_max_dd_ {maximum_fusion_cross_section(true)},
      profile_max_ {maximum_profile_density(
        parameters_, parameters_.cathode_radius_cm /
                       parameters_.anode_radius_cm)}
  {
    if (!(sigma_max_dt_ > 0.0 && sigma_max_dd_ > 0.0)) {
      throw std::runtime_error("Bosch-Hale fits produced no positive cross section");
    }
    if (profile_max_ == 0.0 && parameters_.beam_target_weight > 0.0) {
      throw std::invalid_argument(
        "The neutral density profile is zero throughout the beam-target region");
    }
    strength_ = parameters_.source_strength;
    source_spectrum_bins_mev_ = parameters_.source_spectrum_bins_mev;
    if (!source_spectrum_bins_mev_.empty()) {
      source_spectrum_counts_.resize(source_spectrum_bins_mev_.size() - 1);
    }
  }

  ~IecFusionSource() override
  {
    if (parameters_.source_spectrum_file.empty()) return;
    const std::string path =
      parameters_.source_spectrum_file + "." + std::to_string(getpid()) + ".csv";
    std::ofstream output(path);
    if (!output) {
      std::cerr << "Could not write custom source spectrum to " << path << '\n';
      return;
    }
    std::lock_guard<std::mutex> lock(source_spectrum_mutex_);
    output << std::setprecision(17);
    output << "energy_lower_mev,energy_upper_mev,neutron_count,"
              "underflow_count,overflow_count\n";
    for (size_t index = 0; index < source_spectrum_counts_.size(); ++index) {
      output << source_spectrum_bins_mev_[index] << ','
             << source_spectrum_bins_mev_[index + 1] << ','
             << source_spectrum_counts_[index] << ','
             << source_spectrum_underflow_ << ','
             << source_spectrum_overflow_ << '\n';
    }
  }

  openmc::SourceSite sample(uint64_t* seed) const override
  {
    const double total_zone_weight =
      parameters_.beam_target_weight + parameters_.beam_wall_weight +
      parameters_.beam_beam_weight;
    const double choice = random_unit(seed) * total_zone_weight;
    const bool beam_target = choice < parameters_.beam_target_weight;
    const bool beam_wall =
      !beam_target &&
      choice < parameters_.beam_target_weight +
                 parameters_.beam_wall_weight;
    const char* zone_name =
      beam_target ? "beam-target" : beam_wall ? "beam-wall" : "beam-beam";
    int64_t accepted_fusion_events = 0;
    int64_t accepted_neutron_branches = 0;

    for (int64_t attempt = 0; attempt < 1000000; ++attempt) {
      Event event;
      if (beam_target || beam_wall) {
        event = sample_beam_target_event(seed, parameters_, beam_wall);
      } else {
        event = sample_beam_beam_event(seed, parameters_);
      }

      const double sigma_max =
        event.deuterium_deuterium ? sigma_max_dd_ : sigma_max_dt_;
      const double acceptance = event_acceptance(
        event, parameters_, sigma_max, profile_max_, beam_target);
      if (random_unit(seed) >= acceptance) {
        continue;
      }
      ++accepted_fusion_events;
      // D(d,p)T competes with D(d,n)3He. OpenMC transports only the neutron,
      // so proton-branch events are rejected with the Bosch-Hale branch ratio.
      if (!is_neutron_branch(seed, event)) {
        continue;
      }
      ++accepted_neutron_branches;
      const size_t center_index = std::min(
        static_cast<size_t>(random_unit(seed) * parameters_.reactor_centers.size()),
        parameters_.reactor_centers.size() - 1);
      event.position = event.position + parameters_.reactor_centers[center_index];
      auto site = make_neutron_site(seed, event);
      record_source_energy(site.E / 1.0e6);
      return site;
    }
    throw std::runtime_error(
      "Could not sample a neutron-producing fusion event after 1,000,000 trials; "
      "zone=" + std::string(zone_name) +
      ", accepted fusion events=" + std::to_string(accepted_fusion_events) +
      ", accepted neutron branches=" +
      std::to_string(accepted_neutron_branches) +
      ". Check isotope fractions, zone weights, and density profile");
  }

private:
  void record_source_energy(double energy_mev) const
  {
    if (source_spectrum_counts_.empty()) return;
    std::lock_guard<std::mutex> lock(source_spectrum_mutex_);
    if (energy_mev < source_spectrum_bins_mev_.front()) {
      ++source_spectrum_underflow_;
    } else if (energy_mev > source_spectrum_bins_mev_.back()) {
      ++source_spectrum_overflow_;
    } else {
      auto upper = std::upper_bound(
        source_spectrum_bins_mev_.begin(),
        source_spectrum_bins_mev_.end(),
        energy_mev);
      const size_t index = upper == source_spectrum_bins_mev_.end()
        ? source_spectrum_counts_.size() - 1
        : static_cast<size_t>(
            std::distance(source_spectrum_bins_mev_.begin(), upper) - 1);
      ++source_spectrum_counts_[index];
    }
  }

  Parameters parameters_;
  double sigma_max_dt_;
  double sigma_max_dd_;
  double profile_max_;
  std::vector<double> source_spectrum_bins_mev_;
  mutable std::vector<uint64_t> source_spectrum_counts_;
  mutable uint64_t source_spectrum_underflow_ {};
  mutable uint64_t source_spectrum_overflow_ {};
  mutable std::mutex source_spectrum_mutex_;
};

extern "C" std::unique_ptr<openmc::Source> openmc_create_source(
  std::string parameters)
{
  return std::make_unique<IecFusionSource>(parameters);
}
