#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <mutex>
#include <deque>
#include <chrono>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdio>    // std::snprintf
#include <cstdlib>   // std::getenv

using std::placeholders::_1;
using namespace std::chrono_literals;

struct Sample {
  double t;  // seconds since start
  double a;  // channel A
  double b;  // channel B
};

struct Sample3 {
  double t;
  double a, b, c; // three channels
};

class PlotNode : public rclcpp::Node {
public:
  PlotNode() : rclcpp::Node("plot") {
    topic_of_       = declare_parameter<std::string>("optflow_topic", "/velocitypixel");
    topic_odom_     = declare_parameter<std::string>("odom_topic", "/droneposition/odom");
    topic_imu_      = declare_parameter<std::string>("imu_topic", "/imu");
    topic_position_ = declare_parameter<std::string>("position_topic", "/position");
    history_sec_    = declare_parameter<double>("history_sec", 100.0);
    refresh_hz_     = declare_parameter<double>("refresh_hz", 20.0);
    v_abs_init_     = declare_parameter<double>("v_abs_init", 0.2);
    min_z_          = declare_parameter<double>("min_z", 0.15);

    sub_of_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      topic_of_, 10, std::bind(&PlotNode::ofCb, this, _1));

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      topic_odom_, 50, std::bind(&PlotNode::odomCb, this, _1));

    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      topic_imu_, 100, std::bind(&PlotNode::imuCb, this, _1));

    position_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      topic_position_, 50, std::bind(&PlotNode::positionCb, this, _1));

    const double hz = std::max(1.0, refresh_hz_);
    const auto period_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / hz));
    timer_ = create_wall_timer(period_ns, std::bind(&PlotNode::onTimer, this));

    start_time_ = nowSec();

    if (std::getenv("DISPLAY") == nullptr) {
      RCLCPP_WARN(get_logger(),
        "DISPLAY not set: OpenCV windows may not show. If headless, use Xvfb or publish images to RViz.");
    }

    cv::namedWindow(win_name_of_, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name_of_, 1000, 600);

    cv::namedWindow(win_name_imu_, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name_imu_, 1000, 900); // 3 rows

    cv::namedWindow(win_name_pos_, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name_pos_, 1000, 900); // 3 rows

    RCLCPP_INFO(get_logger(),
      "Plotting:\n"
      "  1) OF vx,vy (1/s) vs Odom vx/z,vy/z (1/s)\n     OF: %s | Odom: %s\n"
      "  2) IMU gyro wx,wy,wz (rad/s) vs Odom angular wx,wy,wz (rad/s)\n     IMU: %s | Odom: %s\n"
      "  3) Position x,y,z vs Odom pose x,y,z\n     Position: %s | Odom: %s",
      topic_of_.c_str(), topic_odom_.c_str(),
      topic_imu_.c_str(), topic_odom_.c_str(),
      topic_position_.c_str(), topic_odom_.c_str());
    RCLCPP_INFO(get_logger(), "Controls: press 'q' or ESC to quit.");
  }

private:
  // ---- Callbacks ----
  void ofCb(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
    double vx = msg->twist.linear.x;
    double vy = msg->twist.linear.y;
    if (!std::isfinite(vx) || !std::isfinite(vy)) return;  // drop NaNs/Infs
    const double t = nowSec() - start_time_;
    std::lock_guard<std::mutex> lk(m_);
    of_buf_.push_back({t, vx, vy});
    trimBuffers();
  }

  void odomCb(const nav_msgs::msg::Odometry::SharedPtr msg) {
    const double t = nowSec() - start_time_;

    // For OF comparison: use linear velocity divided by |Z|
    const auto &tw = msg->twist.twist;
    const auto &pp = msg->pose.pose.position;
    const double Zabs = std::max(min_z_, std::abs(pp.z));
    double vx_over_z = tw.linear.x / Zabs;
    double vy_over_z = tw.linear.y / Zabs;

    // For IMU comparison: angular velocity ground truth
    double wx_gt = tw.angular.x;
    double wy_gt = tw.angular.y;
    double wz_gt = tw.angular.z;

    // For position comparison: odom pose (x,y,z)
    double x_gt = pp.x;
    double y_gt = pp.y;
    double z_gt = pp.z;

    std::lock_guard<std::mutex> lk(m_);
    if (std::isfinite(vx_over_z) && std::isfinite(vy_over_z))
      odom_div_buf_.push_back({t, vx_over_z, vy_over_z});
    if (std::isfinite(wx_gt) && std::isfinite(wy_gt) && std::isfinite(wz_gt))
      odom_w_buf_.push_back({t, wx_gt, wy_gt, wz_gt});
    if (std::isfinite(x_gt) && std::isfinite(y_gt) && std::isfinite(z_gt))
      odom_pos_buf_.push_back({t, x_gt, y_gt, z_gt});

    trimBuffers();
  }

  void imuCb(const sensor_msgs::msg::Imu::SharedPtr msg) {
    double wx = msg->angular_velocity.x;
    double wy = msg->angular_velocity.y;
    double wz = msg->angular_velocity.z;
    if (!std::isfinite(wx) || !std::isfinite(wy) || !std::isfinite(wz)) return;
    const double t = nowSec() - start_time_;
    std::lock_guard<std::mutex> lk(m_);
    imu_w_buf_.push_back({t, wx, wy, wz});
    trimBuffers();
  }

  void positionCb(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
    const double t = nowSec() - start_time_;
    double x = msg->point.x;
    double y = msg->point.y;
    double z = msg->point.z;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return;
    std::lock_guard<std::mutex> lk(m_);
    pos_buf_.push_back({t, x, y, z});
    trimBuffers();
  }

  // ---- Drawing timer ----
  void onTimer() {
    // Copy buffers under lock
    std::deque<Sample> of, od_div;
    std::deque<Sample3> imu_w, od_w, pos, od_pos;
    {
      std::lock_guard<std::mutex> lk(m_);
      of     = of_buf_;
      od_div = odom_div_buf_;
      imu_w  = imu_w_buf_;
      od_w   = odom_w_buf_;
      pos    = pos_buf_;
      od_pos = odom_pos_buf_;
    }

    const double t_now = nowSec() - start_time_;
    const double t_min = std::max(0.0, t_now - history_sec_);
    const double t_max = t_now;

    // --------- Window 1: OF vs Odom/Z (vx, vy) ----------
    renderTwoRowWindow(
      win_name_of_,
      "Top: OF vx [m/s]  &  Odom vx/z [1/s]",
      "Bottom: OF vy [m/s]  &  Odom vy/z [1/s]",
      of, od_div, t_min, t_max,
      /*colorA=*/cv::Scalar(60,200,255),  /*labelA=*/"OF (m/s)",
      /*colorB=*/cv::Scalar(120,255,120), /*labelB=*/"Odom/Z (1/s)"
    );

    // --------- Window 2: IMU gyro vs Odom angular (wx, wy, wz) ----------
    renderThreeRowWindow(
      win_name_imu_,
      "Row1: IMU wx [rad/s]  &  Odom wx [rad/s]",
      "Row2: IMU wy [rad/s]  &  Odom wy [rad/s]",
      "Row3: IMU wz [rad/s]  &  Odom wz [rad/s]",
      imu_w, od_w, t_min, t_max,
      /*colorA=*/cv::Scalar(255,180,60),  /*labelA=*/"IMU ω (rad/s)",
      /*colorB=*/cv::Scalar(120,255,120), /*labelB=*/"Odom ω (rad/s)"
    );

    // --------- Window 3: Position vs Odom Position (x, y, z) ----------
    renderThreeRowWindow(
      win_name_pos_,
      "Row1: Position x [m]  &  Odom x [m]",
      "Row2: Position y [m]  &  Odom y [m]",
      "Row3: Position z [m]  &  Odom z [m]",
      pos, od_pos, t_min, t_max,
      /*colorA=*/cv::Scalar(200,200,255), /*labelA=*/"Position",
      /*colorB=*/cv::Scalar(120,255,120), /*labelB=*/"Odom pose"
    );

    // Key handling (any window)
    int k = cv::waitKey(1);
    if (k == 'q' || k == 'Q' || k == 27) {
      rclcpp::shutdown();
    }
  }

  // ---- Rendering helpers ----
  void renderTwoRowWindow(const std::string& win_name,
                          const std::string& title_top,
                          const std::string& title_bot,
                          const std::deque<Sample>& A,
                          const std::deque<Sample>& B,
                          double t_min, double t_max,
                          const cv::Scalar& colorA, const std::string& labelA,
                          const cv::Scalar& colorB, const std::string& labelB)
  {
    const int W = 1000, H = 600;
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(30,30,30));
    const cv::Rect r_top(60, 30, W - 90, (H/2) - 50);
    const cv::Rect r_bot(60, (H/2) + 20, W - 90, (H/2) - 50);

    auto [a_min, a_max] = computeRange2(A, B, /*use_a=*/true);
    auto [b_min, b_max] = computeRange2(A, B, /*use_a=*/false);

    drawAxes(canvas, r_top, t_min, t_max, a_min, a_max, title_top);
    drawAxes(canvas, r_bot, t_min, t_max, b_min, b_max, title_bot);

    plotSeries2(canvas, r_top, A, t_min, t_max, a_min, a_max, /*use_a=*/true,  colorA);
    plotSeries2(canvas, r_bot, A, t_min, t_max, b_min, b_max, /*use_a=*/false, colorA);
    plotSeries2(canvas, r_top, B, t_min, t_max, a_min, a_max, /*use_a=*/true,  colorB);
    plotSeries2(canvas, r_bot, B, t_min, t_max, b_min, b_max, /*use_a=*/false, colorB);

    drawLegend(canvas, r_top, {labelA, labelB}, {colorA, colorB});
    drawLegend(canvas, r_bot, {labelA, labelB}, {colorA, colorB});

    cv::imshow(win_name, canvas);
  }

  void renderThreeRowWindow(const std::string& win_name,
                            const std::string& title_row1,
                            const std::string& title_row2,
                            const std::string& title_row3,
                            const std::deque<Sample3>& A,
                            const std::deque<Sample3>& B,
                            double t_min, double t_max,
                            const cv::Scalar& colorA, const std::string& labelA,
                            const cv::Scalar& colorB, const std::string& labelB)
  {
    const int W = 1000, H = 900;
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(30,30,30));

    const int pad_top = 30, pad_mid = 20, pad_bot = 30;
    const int h_each = (H - pad_top - pad_bot - 2*pad_mid) / 3;
    cv::Rect r1(60, pad_top,                W - 90, h_each - 10);
    cv::Rect r2(60, pad_top + h_each + pad_mid,      W - 90, h_each - 10);
    cv::Rect r3(60, pad_top + 2*(h_each + pad_mid),  W - 90, h_each - 10);

    auto [r1min, r1max] = computeRange3(A, B, 0);
    auto [r2min, r2max] = computeRange3(A, B, 1);
    auto [r3min, r3max] = computeRange3(A, B, 2);

    drawAxes(canvas, r1, t_min, t_max, r1min, r1max, title_row1);
    drawAxes(canvas, r2, t_min, t_max, r2min, r2max, title_row2);
    drawAxes(canvas, r3, t_min, t_max, r3min, r3max, title_row3);

    plotSeries3(canvas, r1, A, t_min, t_max, r1min, r1max, 0, colorA);
    plotSeries3(canvas, r2, A, t_min, t_max, r2min, r2max, 1, colorA);
    plotSeries3(canvas, r3, A, t_min, t_max, r3min, r3max, 2, colorA);

    plotSeries3(canvas, r1, B, t_min, t_max, r1min, r1max, 0, colorB);
    plotSeries3(canvas, r2, B, t_min, t_max, r2min, r2max, 1, colorB);
    plotSeries3(canvas, r3, B, t_min, t_max, r3min, r3max, 2, colorB);

    drawLegend(canvas, r1, {labelA, labelB}, {colorA, colorB});
    drawLegend(canvas, r2, {labelA, labelB}, {colorA, colorB});
    drawLegend(canvas, r3, {labelA, labelB}, {colorA, colorB});

    cv::imshow(win_name, canvas);
  }

  // ---- Ranges & plotting ----
  std::pair<double,double> computeRange2(const std::deque<Sample>& A,
                                         const std::deque<Sample>& B,
                                         bool use_a_channel) const
  {
    double vmax = v_abs_init_;
    auto consider = [&](double v){ if (std::isfinite(v)) vmax = std::max(vmax, std::abs(v)); };
    for (const auto& s : A) consider(use_a_channel ? s.a : s.b);
    for (const auto& s : B) consider(use_a_channel ? s.a : s.b);
    vmax = std::max(vmax, 0.5);
    return {-vmax, vmax};
  }

  std::pair<double,double> computeRange3(const std::deque<Sample3>& A,
                                         const std::deque<Sample3>& B,
                                         int idx) const
  {
    double vmax = v_abs_init_;
    auto pick = [&](const Sample3& s)->double {
      return (idx==0) ? s.a : (idx==1 ? s.b : s.c);
    };
    auto consider = [&](double v){ if (std::isfinite(v)) vmax = std::max(vmax, std::abs(v)); };
    for (const auto& s : A) consider(pick(s));
    for (const auto& s : B) consider(pick(s));
    vmax = std::max(vmax, 0.5);
    return {-vmax, vmax};
  }

  static double nowSec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
  }

  void trimBuffers() {
    const double t_now = nowSec() - start_time_;
    const double t_keep = t_now - history_sec_;
    auto trim2 = [&](std::deque<Sample>& q){
      while (!q.empty() && q.front().t < t_keep) q.pop_front();
    };
    auto trim3 = [&](std::deque<Sample3>& q){
      while (!q.empty() && q.front().t < t_keep) q.pop_front();
    };
    trim2(of_buf_);
    trim2(odom_div_buf_);
    trim3(imu_w_buf_);
    trim3(odom_w_buf_);
    trim3(pos_buf_);
    trim3(odom_pos_buf_);
  }

  static double mapVal(double v, double vmin, double vmax, int pix_min, int pix_max) {
    if (vmax <= vmin + 1e-9) return pix_min;
    double a = (v - vmin) / (vmax - vmin);
    a = std::clamp(a, 0.0, 1.0);
    return pix_min + a * (pix_max - pix_min);
  }

  void drawAxes(cv::Mat& img, const cv::Rect& roi,
                double tmin, double tmax, double vmin, double vmax,
                const std::string& ylabel)
  {
    cv::rectangle(img, roi, cv::Scalar(180,180,180), 1, cv::LINE_AA);
    int y0 = (int)std::round(mapVal(0.0, vmin, vmax, roi.br().y - 1, roi.tl().y + 1));
    cv::line(img, {roi.x+1, y0}, {roi.x + roi.width - 2, y0}, cv::Scalar(90,90,90), 1, cv::LINE_AA);

    const int nticks_t = 6;
    for (int i=0;i<=nticks_t;i++){
      double tt = tmin + (tmax - tmin) * (double)i / nticks_t;
      int x = (int)std::round(mapVal(tt, tmin, tmax, roi.tl().x + 1, roi.br().x - 1));
      cv::line(img, {x, roi.tl().y}, {x, roi.br().y}, cv::Scalar(50,50,50), 1, cv::LINE_AA);
      char buf[64]; std::snprintf(buf, sizeof(buf), "%.0fs", tt - tmax + history_sec_);
      cv::putText(img, buf, {x-14, roi.br().y + 18}, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180,180,180), 1, cv::LINE_AA);
    }
    const int nticks_y = 4;
    for (int i=0;i<=nticks_y;i++){
      double vv = vmin + (vmax - vmin) * (double)i / nticks_y;
      int y = (int)std::round(mapVal(vv, vmin, vmax, roi.br().y - 1, roi.tl().y + 1));
      cv::line(img, {roi.tl().x, y}, {roi.br().x, y}, cv::Scalar(50,50,50), 1, cv::LINE_AA);
      char buf[64]; std::snprintf(buf, sizeof(buf), "%.2f", vv);
      cv::putText(img, buf, {roi.x - 58, y + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180,180,180), 1, cv::LINE_AA);
    }

    cv::putText(img, ylabel, {roi.x + 6, roi.y - 8}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(220,220,220), 1, cv::LINE_AA);
  }

  void plotSeries2(cv::Mat& img, const cv::Rect& roi, const std::deque<Sample>& buf,
                   double tmin, double tmax, double vmin, double vmax, bool use_a,
                   const cv::Scalar& color)
  {
    if (buf.size() < 2) return;
    cv::Point prevp; bool has_prev = false;
    for (const auto& s : buf) {
      if (s.t < tmin) continue;
      double v = use_a ? s.a : s.b;
      if (!std::isfinite(v)) continue;
      int x = (int)std::round(mapVal(s.t, tmin, tmax, roi.tl().x + 1, roi.br().x - 1));
      int y = (int)std::round(mapVal(v, vmin, vmax, roi.br().y - 1, roi.tl().y + 1));
      cv::Point p(x,y);
      if (has_prev) cv::line(img, prevp, p, color, 2, cv::LINE_AA);
      prevp = p; has_prev = true;
    }
  }

  void plotSeries3(cv::Mat& img, const cv::Rect& roi, const std::deque<Sample3>& buf,
                   double tmin, double tmax, double vmin, double vmax, int idx,
                   const cv::Scalar& color)
  {
    if (buf.size() < 2) return;
    auto pick = [&](const Sample3& s)->double { return (idx==0)?s.a : (idx==1? s.b : s.c); };
    cv::Point prevp; bool has_prev = false;
    for (const auto& s : buf) {
      if (s.t < tmin) continue;
      double v = pick(s);
      if (!std::isfinite(v)) continue;
      int x = (int)std::round(mapVal(s.t, tmin, tmax, roi.tl().x + 1, roi.br().x - 1));
      int y = (int)std::round(mapVal(v, vmin, vmax, roi.br().y - 1, roi.tl().y + 1));
      cv::Point p(x,y);
      if (has_prev) cv::line(img, prevp, p, color, 2, cv::LINE_AA);
      prevp = p; has_prev = true;
    }
  }

  void drawLegend(cv::Mat& img, const cv::Rect& roi,
                  const std::vector<std::string>& names,
                  const std::vector<cv::Scalar>& colors)
  {
    int x = roi.x + 6;
    int y = roi.y + 16;
    for (size_t i = 0; i < names.size(); ++i) {
      cv::rectangle(img, {x, y-10, 16, 8}, colors[i], cv::FILLED, cv::LINE_AA);
      cv::putText(img, names[i], {x+22, y-2}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(220,220,220), 1, cv::LINE_AA);
      y += 18;
    }
  }

  // ---- Params/state ----
  std::string topic_of_, topic_odom_, topic_imu_, topic_position_;
  double history_sec_{200.0};
  double refresh_hz_{20.0};
  double v_abs_init_{3.0};
  double min_z_{0.15};

  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_of_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr position_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Data buffers
  std::deque<Sample>  of_buf_;        // optflow: (vx, vy)
  std::deque<Sample>  odom_div_buf_;  // odom: (vx/z, vy/z)

  std::deque<Sample3> imu_w_buf_;     // IMU gyro: (wx, wy, wz)
  std::deque<Sample3> odom_w_buf_;    // Odom angular: (wx, wy, wz)

  std::deque<Sample3> pos_buf_;       // /position: (x, y, z)
  std::deque<Sample3> odom_pos_buf_;  // odom pose: (x, y, z)

  std::mutex m_;
  double start_time_{0.0};

  const std::string win_name_of_  = "OF (1/s) vs Odom/Z (1/s): vx (top), vy (bottom)";
  const std::string win_name_imu_ = "IMU ω vs Odom ω: wx (row1), wy (row2), wz (row3)";
  const std::string win_name_pos_ = "Position vs Odom Pose: x (row1), y (row2), z (row3)";
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PlotNode>());
  rclcpp::shutdown();
  return 0;
}
