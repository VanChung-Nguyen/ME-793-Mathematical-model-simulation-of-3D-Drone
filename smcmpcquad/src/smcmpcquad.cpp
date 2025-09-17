#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include "nav_msgs/msg/odometry.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include <sensor_msgs/msg/imu.hpp>

#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>

// ACADO
#include "acado_common.h"
#include "acado_auxiliary_functions.h"

using namespace std::chrono_literals;

// ----------------- ACADO aliases -----------------
#define NX          ACADO_NX
#define NXA         ACADO_NXA
#define NU          ACADO_NU
#define N           ACADO_N
#define NOD         ACADO_NOD
#define NY          ACADO_NY
#define NYN         ACADO_NYN
#define NUM_STEPS   1
#define VERBOSE     1

ACADOvariables acadoVariables;
ACADOworkspace acadoWorkspace;

// ----------------- Helpers -----------------
template <typename T>
T clamp(T v, T mn, T mx) { return std::max(mn, std::min(mx, v)); }

// ----------------- PIDXY -----------------
class PIDXY {
public:
  PIDXY(double kpx, double kix, double kdx,
        double kpy, double kiy, double kdy)
      : kpx_(kpx), kix_(kix), kdx_(kdx),
        kpy_(kpy), kiy_(kiy), kdy_(kdy) {}

  std::pair<double,double> calculatexy(double setpointx, double setpointy,
                                       double pvx, double pvy, double dt) {
    double ex = setpointx - pvx;
    double ey = setpointy - pvy;

    integral_ex_ += ex * dt;
    integral_ey_ += ey * dt;
    integral_ex_ = clamp(integral_ex_, -imax_, imax_);
    integral_ey_ = clamp(integral_ey_, -imax_, imax_);

    // Derivative terms
    double dex = (dt > 1e-5) ? (ex - prev_ex_) / dt : 0.0;
    double dey = (dt > 1e-5) ? (ey - prev_ey_) / dt : 0.0;
    prev_ex_ = ex;
    prev_ey_ = ey;

    // PID control law
    double thetar = kpx_*ex + kix_*integral_ex_ + kdx_*dex;
    double phir   = -(kpy_*ey + kiy_*integral_ey_ + kdy_*dey);

    return { clamp(phir, -0.2, 0.2), clamp(thetar, -0.2, 0.2) };
  }

private:
  double kpx_, kix_, kdx_;
  double kpy_, kiy_, kdy_;
  double prev_ex_ = 0.0, prev_ey_ = 0.0;
  double integral_ex_ = 0.0, integral_ey_ = 0.0;
  double imax_ = 1.0; // anti-windup limit

};

// ----------------- SMCMPC Node -----------------
class SMCMPC : public rclcpp::Node {
public:
  SMCMPC(): Node("SMCMPC"), controlxy_(0.05, 0.0, 0.1, 0.05, 0.0, 0.1), xr_(0.0), yr_(0.0), zr_(1.0), psir_(0.0), x_(0.0), y_(0.0), z_(0.2), phi_(0.0), theta_(0.0), psi_(0.0), pretime_(this->get_clock()->now())
  {
    // Publishers
    propvel_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/prop_vel", 10);
    control_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/controllaw", 10);

    // Subscribers
    trajectory_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/trajectory", 10, std::bind(&SMCMPC::trajectoryCallback, this, std::placeholders::_1));

    position_sub_  = create_subscription<geometry_msgs::msg::PointStamped>(
      "/position", 10, std::bind(&SMCMPC::positionCallback, this, std::placeholders::_1));

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu", rclcpp::SensorDataQoS(), std::bind(&SMCMPC::imuCallback, this, std::placeholders::_1));
    
  }

private:
  // ---------- Callbacks ----------
  void trajectoryCallback (const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (msg->data.size() != 3) {
      RCLCPP_WARN(get_logger(), "Trajectory msg must be [xr, yr, zr]; got %zu", msg->data.size());
      return;
    }
    xr_ = msg->data[0];
    yr_ = msg->data[1];
    zr_ = msg->data[2];
  }

  void positionCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
    x_ = msg->point.x;
    y_ = msg->point.y;
    z_ = msg->point.z;
  }

  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    rclcpp::Time timenow = this->get_clock()->now();
    double dt = (timenow - pretime_).seconds();
    pretime_ = timenow;

    time = time + dt; 

 
    xr_ = 3 * sin(0.04 * (time));
    yr_ = 3 * sin(0.08 * (time));
    zr_ = 1.0;

    //xr_ = 0;
    //yr_ = 0;
    //zr_ = 1.0;
    
    const double wx = msg->angular_velocity.x - gyro_bias_x_;
    const double wy = msg->angular_velocity.y - gyro_bias_y_;
    const double wz = msg->angular_velocity.z - gyro_bias_z_;

    const double wnorm = std::sqrt(wx*wx + wy*wy + wz*wz);
    tf2::Quaternion dq;
    if (wnorm > 1e-9) {
      const double angle = wnorm * dt;
      const double half  = 0.5 * angle;
      const double s     = std::sin(half) / wnorm;
      dq.setW(std::cos(half));
      dq.setX(wx * s);
      dq.setY(wy * s);
      dq.setZ(wz * s);
    } else {
      dq.setW(1.0);
      dq.setX(0.5 * wx * dt);
      dq.setY(0.5 * wy * dt);
      dq.setZ(0.5 * wz * dt);
    }
    // body rates ⇒ q_k = q_{k-1} * dq
    q_hat_ = q_hat_ * dq;
    q_hat_.normalize();

    // Extract RPY (rad)
    tf2::Matrix3x3(q_hat_).getRPY(phi_, theta_, psi_);

    // Angular rates (rad/s)
    const double phid   = wx;
    const double thetad = wy;
    const double psid   = wz;

    double xd = (x_ - prev_x_) / dt;
    double yd = (y_ - prev_y_) / dt;
    double zd = (z_ - prev_z_) / dt;
    prev_x_ = x_; prev_y_ = y_; prev_z_ = z_;

    // ===== 3) Reference shaping =====
    const double xrd = 0;
    const double yrd = 0;
    const double zrd = 0;
    const double xrdd = 0.0, yrdd = 0.0, zrdd = 0.0;

    // Desired roll/pitch from XY controller
    auto [phir, thetar] = controlxy_.calculatexy(xr_, yr_, x_, y_, dt);
    const double phird   = (phir   - prephir_)   / dt;
    const double thetard = (thetar - prethetar_) / dt;
    const double psird   = (psir_  - prepsir_)   / dt;
    prephir_ = phir; prethetar_ = thetar; prepsir_ = psir_;

    // ===== 4) Attitude SMC =====
    const double ixx = 0.0785, iyy = 0.0785, izz = 0.105;

    const double cphi_ctrl = 12.5, ctheta_ctrl = 12.5, cpsi_ctrl = 12.5;
    const double Kx = 32.5, Ky = 32.5, Kz = 27.5;

    const double fphi   = psid * thetad * (iyy - izz) / ixx;
    const double ftheta = psid * phid   * (izz - ixx) / iyy;
    const double fpsi   = phid * thetad * (ixx - iyy) / izz;

    const double bphi = 1.0/ixx, btheta = 1.0/iyy, bpsi = 1.0/izz;
    const double Kphi = 1.75, Ktheta = 1.75, Kpsi = 0.75;

    const double scphi   = cphi_ctrl   * (phi_   - phir)   + (phid   - phird);
    const double sctheta = ctheta_ctrl * (theta_ - thetar) + (thetad - thetard);
    const double scpsi   = cpsi_ctrl   * (psi_   - psir_)  + (psid   - psird);

    const double satphi   = clamp(scphi,   -0.1, 0.1);
    const double sattheta = clamp(sctheta, -0.1, 0.1);
    const double satpsi   = clamp(scpsi,   -0.1, 0.1);

    const double U2s = (-Ky * cphi_ctrl   * phi_   - (cphi_ctrl   + Ky) * phid   + Ky * cphi_ctrl   * phir   + (cphi_ctrl   + Ky) * phird   - fphi   - Kphi   * satphi  ) / bphi;
    const double U3s = (-Kx * ctheta_ctrl * theta_ - (ctheta_ctrl + Kx) * thetad + Kx * ctheta_ctrl * thetar + (ctheta_ctrl + Kx) * thetard - ftheta - Ktheta * sattheta) / btheta;
    const double U4s = (-Kz * cpsi_ctrl   * psi_   - (cpsi_ctrl   + Kz) * psid   + Kz * cpsi_ctrl   * psir_  + (cpsi_ctrl   + Kz) * psird   - fpsi   - Kpsi   * satpsi  ) / bpsi;

    const double U2smc = clamp(U2s, -3.5, 3.5);
    const double U3smc = clamp(U3s, -3.5, 3.5);
    const double U4smc = clamp(U4s, -3.5, 3.5);

    // ===== 5) Translational SMC =====
    const double m = 1.85, g = 9.80;
    const double Kdx = 0.00000267, Kdy = 0.00000267, Kdz = 0.00000625;
    const double cx = 0.015, cy = 0.015, cz = 0.375;

    const double cphi = std::cos(phi_), sphi = std::sin(phi_);
    const double ctheta = std::cos(theta_), stheta = std::sin(theta_);
    const double cpsi = std::cos(psi_), spsi = std::sin(psi_);

    const double fx = -Kdx * xd / m;
    const double fy = -Kdy * yd / m;
    const double fz = (-Kdz * zd - m * g) / m;

    const double bx = (1.0/m) * ( spsi * sphi + cpsi * stheta * cphi);
    const double by = (1.0/m) * ( spsi * stheta * cphi - cpsi * sphi);
    const double bz = (1.0/m) * ( ctheta * cphi);

    const double sx = cx * (xr_ - x_) + (xrd - xd);
    const double sy = cy * (yr_ - y_) + (yrd - yd);
    const double sz = cz * (zr_ - z_) + (zrd - zd);

    double ueqx = (cx * (xrd - xd) + xrdd - fx) / bx;
    double ueqy = (cy * (yrd - yd) + yrdd - fy) / by;
    double ueqz = (cz * (zrd - zd) + zrdd - fz) / bz;

    ueqx = clamp(ueqx, -0.15, 0.15);
    ueqy = clamp(ueqy, -0.15, 0.15);

    const double lam1 = 0.05, lam2 = 0.05, Ka = 2.75, eta = 0.175;
    const double s3 = lam2 * lam1 * sx + lam2 * sy + sz;
    const double sats3 = clamp(s3, -0.1, 0.1);

    const double usw = -(lam2*lam1*bx*(ueqy + ueqz) + lam2*by*(ueqx + ueqz) + bz*(ueqx + ueqy) - Ka*s3 - eta*sats3)
                       / (lam2*lam1*bx + lam2*by + bz);
    const double Uz = ueqx + ueqy + ueqz + usw;
    const double U1smc = clamp(Uz, 14.5, 22.5);

    // ===== 6) ACADO NMPC (as in your original) =====
    unsigned int i;
    std::memset(&acadoWorkspace, 0, sizeof(acadoWorkspace));
    std::memset(&acadoVariables, 0, sizeof(acadoVariables));
    acado_initializeSolver();

    for (i = 0; i < N + 1; ++i) {
      acadoVariables.x[i*NX +  0] = x_;
      acadoVariables.x[i*NX +  1] = y_;
      acadoVariables.x[i*NX +  2] = z_;
      acadoVariables.x[i*NX +  3] = phi_;
      acadoVariables.x[i*NX +  4] = theta_;
      acadoVariables.x[i*NX +  5] = psi_;
      acadoVariables.x[i*NX +  6] = xd;
      acadoVariables.x[i*NX +  7] = yd;
      acadoVariables.x[i*NX +  8] = zd;
      acadoVariables.x[i*NX +  9] = phid;
      acadoVariables.x[i*NX + 10] = thetad;
      acadoVariables.x[i*NX + 11] = psid;
    }

    for (i = 0; i < N; ++i) {
      acadoVariables.y[i*NY + 0] = zr_;
      acadoVariables.y[i*NY + 1] = phir;
      acadoVariables.y[i*NY + 2] = thetar;
      acadoVariables.y[i*NY + 3] = psir_;
      acadoVariables.y[i*NY + 4] = U1smc;
      acadoVariables.y[i*NY + 5] = U2smc;
      acadoVariables.y[i*NY + 6] = U3smc;
      acadoVariables.y[i*NY + 7] = U4smc;
    }

    acadoVariables.yN[0] = zr_;
    acadoVariables.yN[1] = phir;
    acadoVariables.yN[2] = thetar;
    acadoVariables.yN[3] = psir_;

    for (i = 0; i < NX; ++i) acadoVariables.x0[i] = acadoVariables.x[NX + i];

    acado_preparationStep();
    acado_feedbackStep();
    acado_shiftStates(2, 0, 0);
    acado_shiftControls(0);
    real_t* u = acado_getVariablesU();

    // Extract controls over horizon
    std::vector<std::vector<double>> control_variables;
    control_variables.reserve(N);
    for (int k = 0; k < N; ++k) {
      std::vector<double> row;
      for (int j = 0; j < NU; ++j) row.push_back(static_cast<double>(u[k*NU + j]));
      control_variables.push_back(row);
    }

    // Use NMPC control at stage 1 (as in your code)
    double U1 = control_variables[1][0];
    double U2 = control_variables[1][1];
    double U3 = control_variables[1][2];
    double U4 = control_variables[1][3];

    // ===== 7) Actuator mapping =====
    double kt = 0.00025, kd = 0.000075, l = 0.159;

    double w12 = (U1*kd*l - U2*kd - U3*kd + U4*kt*l) / (4*kd*kt*l);
    double w22 = (U1*kd*l + U2*kd - U3*kd - U4*kt*l) / (4*kd*kt*l);
    double w32 = (U1*kd*l - U2*kd + U3*kd - U4*kt*l) / (4*kd*kt*l);
    double w42 = (U1*kd*l + U2*kd + U3*kd + U4*kt*l) / (4*kd*kt*l);

    w12 = std::max(0.0, w12); w22 = std::max(0.0, w22);
    w32 = std::max(0.0, w32); w42 = std::max(0.0, w42);

    double w1 = -std::sqrt(w12);
    double w2 =  std::sqrt(w22);
    double w3 =  std::sqrt(w32);
    double w4 = -std::sqrt(w42);

    // ===== 8) Publish =====
    std_msgs::msg::Float64MultiArray prop_vel_msg; prop_vel_msg.data = {w1, w2, w3, w4};
    propvel_pub_->publish(prop_vel_msg);

    std_msgs::msg::Float64MultiArray control_msg;  control_msg.data = {U1, U2, U3, U4};
    control_pub_->publish(control_msg);
  }

  // ---------- Members ----------
  // pubs/subs
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr propvel_pub_, control_pub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr trajectory_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr position_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

  rclcpp::Time pretime_;

  double xr_=0.0, yr_=0.0, zr_=0.0, psir_=0.0;

  bool got_pos_ = false;
  double x_=0.0, y_=0.0, z_=0.0;
  double prev_x_=0.0, prev_y_=0.0, prev_z_=0.0;

  tf2::Quaternion q_hat_{0,0,0,1};
  double phi_=0.0, theta_=0.0, psi_=0.0; 
  double gyro_bias_x_=0.0, gyro_bias_y_=0.0, gyro_bias_z_=0.0;

  double prephi_, pretheta_, prepsi_;
  double prephir_=0.0, prethetar_=0.0, prepsir_=0.0;

  PIDXY controlxy_;
  double time = 0;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto controller = std::make_shared<SMCMPC>();
  rclcpp::spin(controller);
  rclcpp::shutdown();
  return 0;
}
