#include <cmath>
#include "openmc/random_lcg.h"
#include "openmc/source.h"

// Mass constants in MeV/c^2
constexpr double MASS_D = 1875.612;
constexpr double MASS_T = 2808.921;
constexpr double MASS_N = 939.565;
constexpr double MASS_ALPHA = 3727.379;
constexpr double MASS_HE3 = 2808.391;

// Reaction Q-values in MeV
constexpr double Q_DT = 17.589;
constexpr double Q_DD = 3.269;

// IEC Geometry & Parameters
constexpr double R_ANODE = 4.05;   // cm (outer grounded vessel)
constexpr double R_CATHODE = 0.73; // cm (inner grid / virtual anode boundary)
constexpr double V_APPLIED = 0.09; // MeV (e.g., 100 kV)

// Helper: Evaluates empirical cross sections (Replace with Bosch-Hale polynomial fit)
double evaluate_cross_section(double E_MeV, bool is_DT) {
    if (E_MeV <= 0.0) return 0.0;
    return std::exp(-1.0 / std::sqrt(E_MeV)); // Placeholder dummy scaling
}

// Helper: Sample isotropic direction
void sample_isotropic(uint64_t* seed, double dir[3]) {
    double costheta = 2.0 * openmc::prn(seed) - 1.0;
    double sintheta = std::sqrt(1.0 - costheta * costheta);
    double phi = 2.0 * M_PI * openmc::prn(seed);
    dir[0] = sintheta * std::cos(phi);
    dir[1] = sintheta * std::sin(phi);
    dir[2] = costheta;
}

extern "C" openmc::SourceSite sample_source(uint64_t* seed) {
    openmc::SourceSite site;
    site.wgt = 1.0;
    site.delayed_group = 0;

    // Define reaction branch (e.g., 99% DT, 1% DD)
    bool is_DT = (openmc::prn(seed) < 0.99);
    
    double m1 = MASS_D;
    double m2 = is_DT ? MASS_T : MASS_D;
    double m3 = MASS_N;
    double m4 = is_DT ? MASS_ALPHA : MASS_HE3;
    double Q  = is_DT ? Q_DT : Q_DD;

    // Determine Interaction Zone
    // In a full model, calculate these dynamically by integrating the cross section 
    // over the background gas density and grid transparency.
    double rnd = openmc::prn(seed);
    double p_beam_target = 0.60;
    double p_beam_wall   = 0.30;
    // Remaining 10% is beam-beam in the virtual anode
    
    double E1 = 0.0, E2 = 0.0;
    double dir1[3] = {0}, dir2[3] = {0};
    double x = 0, y = 0, z = 0;

    if (rnd < p_beam_target) {
        // 1. Beam-Target in Plasma Volume (Rejection Sampling)
        double sigma_max = evaluate_cross_section(V_APPLIED, is_DT);
        bool accepted = false;
        
        while (!accepted) {
            // Sample radius uniformly within the cylindrical volume
            double r = R_ANODE * std::sqrt(openmc::prn(seed)); 
            
            // Simplified linear electric field approximation for local energy
            E1 = V_APPLIED * (r / R_ANODE); 
            
            // Rejection test against cross section at this specific voltage
            double sigma = evaluate_cross_section(E1, is_DT);
            if (openmc::prn(seed) < (sigma / sigma_max)) {
                double theta = 2.0 * M_PI * openmc::prn(seed);
                x = r * std::cos(theta);
                y = r * std::sin(theta);
                z = 20.0 * (openmc::prn(seed) - 0.5); // +/- 10cm axial length
                
                // Ion 1 moves inward radially towards the cathode
                double norm = std::sqrt(x*x + y*y);
                dir1[0] = -x / norm; dir1[1] = -y / norm; dir1[2] = 0.0;
                
                // Target is thermal background gas (effectively stationary)
                E2 = 0.0; 
                accepted = true;
            }
        }
    } else if (rnd < p_beam_target + p_beam_wall) {
        // 2. Beam-Wall on the Grid Surface
        double r = R_CATHODE;
        double theta = 2.0 * M_PI * openmc::prn(seed);
        x = r * std::cos(theta);
        y = r * std::sin(theta);
        z = 20.0 * (openmc::prn(seed) - 0.5);
        
        E1 = V_APPLIED; // Max energy achieved at the grid
        E2 = 0.0;       // Target embedded in the grid is stationary
        
        double norm = std::sqrt(x*x + y*y);
        dir1[0] = -x / norm; dir1[1] = -y / norm; dir1[2] = 0.0;
    } else {
        // 3. Beam-Beam inside the Virtual Anode (Core)
        // Volumetric sampling inside the central core
        double r = R_CATHODE * std::cbrt(openmc::prn(seed)); 
        double theta = 2.0 * M_PI * openmc::prn(seed);
        double phi = std::acos(2.0 * openmc::prn(seed) - 1.0);
        
        x = r * std::sin(phi) * std::cos(theta);
        y = r * std::sin(phi) * std::sin(theta);
        z = r * std::cos(phi);
        
        E1 = V_APPLIED; // Assuming negligible potential drop inside the grid
        E2 = V_APPLIED; 
        
        // Two ions interacting. For simplicity, assume they collide head-on isotropically.
        sample_isotropic(seed, dir1);
        dir2[0] = -dir1[0]; dir2[1] = -dir1[1]; dir2[2] = -dir1[2]; 
    }

    site.r.x = x; site.r.y = y; site.r.z = z;

    // --- Kinematics Transformation ---
    
    // Velocities in units of c
    double v1 = std::sqrt(2.0 * E1 / m1);
    double v2 = std::sqrt(2.0 * E2 / m2);
    
    double vec_v1[3] = {v1 * dir1[0], v1 * dir1[1], v1 * dir1[2]};
    double vec_v2[3] = {v2 * dir2[0], v2 * dir2[1], v2 * dir2[2]};
    
    // COM Velocity
    double vec_vcm[3];
    for (int i=0; i<3; i++) {
        vec_vcm[i] = (m1 * vec_v1[i] + m2 * vec_v2[i]) / (m1 + m2);
    }
    
    // Relative kinetic energy
    double v_rel_sq = 0.0;
    for (int i=0; i<3; i++) {
        double diff = vec_v1[i] - vec_v2[i];
        v_rel_sq += diff * diff;
    }
    double E_kin_cm = 0.5 * (m1 * m2 / (m1 + m2)) * v_rel_sq;
    
    // Total available energy in COM
    double E_total_cm = E_kin_cm + Q;
    
    // Neutron energy and velocity in COM
    double En_cm = E_total_cm * (m4 / (m3 + m4));
    double vn_cm = std::sqrt(2.0 * En_cm / m3); 
    
    double u_cm[3];
    sample_isotropic(seed, u_cm);
    
    // Transform neutron trajectory back to the Lab Frame
    double vec_vn_lab[3];
    double vn_lab_sq = 0.0;
    for (int i=0; i<3; i++) {
        vec_vn_lab[i] = vn_cm * u_cm[i] + vec_vcm[i];
        vn_lab_sq += vec_vn_lab[i] * vec_vn_lab[i];
    }
    
    // Output Lab Energy (OpenMC requires eV)
    site.E = 0.5 * m3 * vn_lab_sq * 1e6; 
    
    // Output Lab Direction (Normalized)
    double vn_lab_mag = std::sqrt(vn_lab_sq);
    site.u.x = vec_vn_lab[0] / vn_lab_mag;
    site.u.y = vec_vn_lab[1] / vn_lab_mag;
    site.u.z = vec_vn_lab[2] / vn_lab_mag;
    
    return site;
}