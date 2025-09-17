// optical_flow_velocity_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <limits>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

using std::placeholders::_1;

template <typename T>
static inline T clamp(T v, T lo, T hi) { return std::max(lo, std::min(hi, v)); }

class OpticalFlowVelocityNode : public rclcpp::Node {
public:
  OpticalFlowVelocityNode() : Node("optical_flow_velocity_node") {
    // Ensure last_range_time_ is initialized to "now" for grace logic.
    last_range_time_ = this->get_clock()->now();

    // Topics / frame
    image_topic_       = declare_parameter<std::string>("image_topic", "/oakd_rgb/image_raw");
    camera_info_topic_ = declare_parameter<std::string>("camera_info_topic", "/oakd_rgb/camera_info");
    range_topic_       = declare_parameter<std::string>("range_topic", "/lidar_distance");
    imu_topic_         = declare_parameter<std::string>("imu_topic", "/imu");
    publish_frame_id_  = declare_parameter<std::string>("frame_id", "camera_rgb_frame");

    // Sensing
    use_range_topic_   = declare_parameter<bool>("use_range_topic", true);
    height_fixed_      = declare_parameter<double>("height_fixed", 1.0);

    // Processing toggles
    proc_scale_        = declare_parameter<double>("proc_scale", 0.5);
    do_undistort_      = declare_parameter<bool>("do_undistort", false);
    do_clahe_          = declare_parameter<bool>("do_clahe", false);
    do_fb_check_       = declare_parameter<bool>("do_fb_check", false);
    derotate_with_imu_ = declare_parameter<bool>("derotate_with_imu", true);
    compensate_rot_velocity_ = declare_parameter<bool>("compensate_rot_velocity", false);

    // Features / flow (defaults tuned for robustness)
    min_features_         = declare_parameter<int>("min_features", 120);
    max_features_         = declare_parameter<int>("max_features", 400);
    quality_level_        = declare_parameter<double>("quality_level", 0.01);
    min_distance_         = declare_parameter<double>("min_distance", 8.0);
    ransac_reproj_thresh_ = declare_parameter<double>("ransac_reproj_thresh", 3.0);
    fb_max_error_px_      = declare_parameter<double>("fb_max_error_px", 1.0);

    // LK + gating
    lk_levels_  = declare_parameter<int>("lk_levels", 5);
    lk_win_px_  = declare_parameter<int>("lk_win_px", 31);
    min_inliers_publish_ = declare_parameter<int>("min_inliers_publish", 15);
    max_shift_px_        = declare_parameter<double>("max_shift_px", 8.0);
    max_dt_              = declare_parameter<double>("max_dt", 0.5);
    flow_eps_px_         = declare_parameter<double>("flow_eps_px", 0.10);

    // Robustness helpers
    range_grace_sec_     = declare_parameter<double>("range_grace_sec", 0.3);
    always_integrate_raw_= declare_parameter<bool>("always_integrate_raw", false);
    raw_vmax_mps_        = declare_parameter<double>("raw_vmax_mps", 5.0);
    reseed_every_        = declare_parameter<int>("reseed_every", 20);

    // Optional clamp for even valid velocities (protects against Z/dt jitter spikes)
    max_valid_v_mps_     = declare_parameter<double>("max_valid_v_mps", 15.0);

    // Subs
    img_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(),
        std::bind(&OpticalFlowVelocityNode::imageCb, this, _1));

    caminfo_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, 10,
        std::bind(&OpticalFlowVelocityNode::camInfoCb, this, _1));

    if (use_range_topic_) {
      range_sub_ = create_subscription<sensor_msgs::msg::Range>(
          range_topic_, rclcpp::SensorDataQoS(),
          std::bind(&OpticalFlowVelocityNode::rangeCb, this, _1));
    }

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, rclcpp::SensorDataQoS(),
        std::bind(&OpticalFlowVelocityNode::imuCb, this, _1));

    // Pubs
    vel_pub_      = create_publisher<geometry_msgs::msg::TwistStamped>("/velocity", 10);
    pixelvel_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("/velocitypixel", 10);
    position_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("/position", 10);

    RCLCPP_INFO(get_logger(),
      "OF robust: img=%s cam_info=%s range=%s imu=%s frame=%s (scale=%.2f undistort=%d clahe=%d fb=%d imu-derot=%d rot-comp=%d "
      "lk=%dx%d inliers>=%d shift<=%.1f max_dt=%.2f)",
      image_topic_.c_str(), camera_info_topic_.c_str(), range_topic_.c_str(), imu_topic_.c_str(),
      publish_frame_id_.c_str(), proc_scale_, do_undistort_, do_clahe_, do_fb_check_, derotate_with_imu_, compensate_rot_velocity_,
      lk_levels_, lk_win_px_, min_inliers_publish_, max_shift_px_, max_dt_);
  }

private:
  // ------- Callbacks -------
  void camInfoCb(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    if (msg->k.size() < 9) return;
    fx_full_ = msg->k[0]; fy_full_ = msg->k[4];
    cx_full_ = msg->k[2]; cy_full_ = msg->k[5];
    full_size_ = cv::Size((int)msg->width, (int)msg->height);

    fx_ = fx_full_ * proc_scale_;
    fy_ = fy_full_ * proc_scale_;
    cx_ = cx_full_ * proc_scale_;
    cy_ = cy_full_ * proc_scale_;
    proc_size_ = cv::Size((int)std::round(full_size_.width*proc_scale_),
                          (int)std::round(full_size_.height*proc_scale_));

    if (do_undistort_) {
      cv::Mat K = (cv::Mat_<double>(3,3) << fx_full_, 0, cx_full_, 0, fy_full_, cy_full_, 0, 0, 1);
      cv::Mat D;
      if (!msg->d.empty()) {
        D = cv::Mat((int)msg->d.size(), 1, CV_64F);
        for (size_t i=0;i<msg->d.size();++i) D.at<double>((int)i) = msg->d[i];
      } else {
        D = cv::Mat::zeros(5,1,CV_64F);
      }
      cv::Mat Kscaled = (cv::Mat_<double>(3,3) << fx_, 0, cx_, 0, fy_, cy_, 0,0,1);
      cv::initUndistortRectifyMap(K, D, cv::Mat::eye(3,3,CV_64F), Kscaled, proc_size_,
                                  CV_32FC1, map1_, map2_);
      have_maps_ = true;
    } else {
      have_maps_ = false;
      map1_.release(); map2_.release();
    }

    Kproc_ = (cv::Mat_<double>(3,3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
    cv::invert(Kproc_, Kproc_inv_);
    has_cam_info_ = (fx_>0 && fy_>0);
  }

  void rangeCb(const sensor_msgs::msg::Range::SharedPtr msg) {
    if (!std::isfinite(msg->range)) return;
    if (msg->range < msg->min_range || msg->range > msg->max_range) return;
    height_m_ = msg->range; have_valid_range_ = true;
    last_range_time_ = now();
  }

  void imuCb(const sensor_msgs::msg::Imu::SharedPtr msg) {
    wx_ = msg->angular_velocity.x;
    wy_ = msg->angular_velocity.y;
    wz_ = msg->angular_velocity.z;
    have_imu_ = true;
  }

  // ------- Robust helpers -------
  static inline double robustMedian(std::vector<double> v) {
    if (v.empty()) return 0.0;
    size_t n = v.size()/2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    double m = v[n];
    if (v.size()%2==0 && n>0) { std::nth_element(v.begin(), v.begin()+n-1, v.end()); m = 0.5*(m + v[n-1]); }
    return m;
  }

  static inline double robustMedianDelta(const std::vector<cv::Point2f>& a,
                                         const std::vector<cv::Point2f>& b,
                                         int axis /*0=x,1=y*/) {
    std::vector<double> v; v.reserve(a.size());
    if (axis==0) for (size_t i=0;i<a.size();++i) v.push_back(b[i].x - a[i].x);
    else         for (size_t i=0;i<a.size();++i) v.push_back(b[i].y - a[i].y);
    return robustMedian(std::move(v));
  }

  static inline void dxdy(const std::vector<cv::Point2f>& a,
                          const std::vector<cv::Point2f>& b,
                          std::vector<double>& dx, std::vector<double>& dy) {
    dx.clear(); dy.clear(); dx.reserve(a.size()); dy.reserve(a.size());
    for (size_t i=0;i<a.size();++i) { dx.push_back(b[i].x-a[i].x); dy.push_back(b[i].y-a[i].y); }
  }

  static inline double MAD(std::vector<double> vals) {
    if (vals.empty()) return 0.0;
    double med = robustMedian(vals);
    for (double &x : vals) x = std::abs(x - med);
    return 1.4826 * robustMedian(vals);
  }

  static void trim_outliers(std::vector<cv::Point2f>& a,
                            std::vector<cv::Point2f>& b) {
    if (a.size() != b.size() || a.empty()) return;
    std::vector<double> dx,dy; dxdy(a,b,dx,dy);
    double madx = std::max(1e-6, MAD(dx));
    double mady = std::max(1e-6, MAD(dy));
    double medx = robustMedian(dx);
    double medy = robustMedian(dy);
    std::vector<cv::Point2f> a2; a2.reserve(a.size());
    std::vector<cv::Point2f> b2; b2.reserve(b.size());
    for (size_t i=0;i<dx.size();++i) {
      if (std::abs(dx[i]-medx) < 3*madx && std::abs(dy[i]-medy) < 3*mady) { a2.push_back(a[i]); b2.push_back(b[i]); }
    }
    a.swap(a2); b.swap(b2);
  }

  static int countInliersFlexible(const cv::Mat& inliers, int expected) {
    if (inliers.empty()) return expected;
    if (inliers.rows == expected && inliers.cols == 1) {
      return cv::countNonZero(inliers);
    }
    // Some OpenCV builds return 1xN; reshape to Nx1 if possible.
    cv::Mat flat = inliers.reshape(1, inliers.total());
    return cv::countNonZero(flat);
  }

  // ------- Main -------
  void imageCb(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (!has_cam_info_) return;

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cv_bridge MONO8 failed: %s", e.what());
      return;
    }

    const cv::Mat &src_mono = cv_ptr->image;
    if (src_mono.empty()) return;

    if (proc_size_.width == 0 || proc_size_.height == 0) proc_size_ = src_mono.size();

    if (proc_scale_ != 1.0) cv::resize(src_mono, gray_proc_, proc_size_, 0,0, cv::INTER_AREA);
    else                    gray_proc_ = src_mono;

    if (do_undistort_ && have_maps_) cv::remap(gray_proc_, gray_proc_ud_, map1_, map2_, cv::INTER_LINEAR);
    else                              gray_proc_ud_ = gray_proc_;
    if (do_clahe_) { if (!clahe_) clahe_ = cv::createCLAHE(3.0, cv::Size(8,8)); clahe_->apply(gray_proc_ud_, curr_img_); }
    else curr_img_ = gray_proc_ud_;

    // Height with grace window
    const double Z = heightWithGrace();
    if (!(Z > 0.02) || !std::isfinite(Z)) { resetFrameState(msg->header.stamp); return; }

    const rclcpp::Time t_now = msg->header.stamp;
    if (prev_time_.nanoseconds() == 0) { initFeatures(curr_img_); curr_img_.copyTo(prev_img_); prev_time_ = t_now; return; }

    const double dt = (t_now - prev_time_).seconds();
    if (!(dt > 0.0) || dt > max_dt_ || !std::isfinite(dt)) {
      RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000, "Skip frame: dt=%.3f (max_dt=%.3f)", dt, max_dt_);
      resetFrameState(t_now);
      return;
    }

    // IMU derotation (optional)
    cv::Mat prev_proc = prev_img_;
    if (derotate_with_imu_ && have_imu_) {
      cv::Mat rvec = (cv::Mat_<double>(3,1) << wx_*dt, wy_*dt, wz_*dt);
      cv::Mat R; cv::Rodrigues(rvec, R);
      cv::Mat H = Kproc_ * R * Kproc_inv_;
      cv::warpPerspective(prev_img_, warped_prev_, H, prev_img_.size(),
                          cv::INTER_LINEAR, cv::BORDER_REPLICATE);
      prev_proc = warped_prev_;
    }

    if ((int)prev_pts_.size() < min_features_) initFeatures(prev_proc);
    if (prev_pts_.empty()) { resetFrameState(t_now); return; }

    // Lucas–Kanade
    const cv::Size lk_win(lk_win_px_, lk_win_px_);
    next_pts_.clear(); status_.clear(); err_.clear();
    cv::calcOpticalFlowPyrLK(prev_proc, curr_img_, prev_pts_, next_pts_, status_, err_,
                             lk_win, lk_levels_,
                             cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-3),
                             0, 1e-4);

    std::vector<cv::Point2f> s_prev; s_prev.reserve(next_pts_.size());
    std::vector<cv::Point2f> s_next; s_next.reserve(next_pts_.size());
    for (size_t i=0;i<next_pts_.size();++i)
      if (status_[i]) { s_prev.push_back(prev_pts_[i]); s_next.push_back(next_pts_[i]); }
    if ((int)s_prev.size() < 10) { initFeatures(curr_img_); resetFrameState(t_now); return; }

    // FB check when flow small or requested
    const double mdu0 = robustMedianDelta(s_prev, s_next, 0);
    const double mdv0 = robustMedianDelta(s_prev, s_next, 1);
    const double flow_mag0 = std::hypot(mdu0, mdv0);
    const bool do_fb = do_fb_check_ || (flow_mag0 < flow_eps_px_);

    p0_.clear(); p1_.clear();
    if (do_fb) {
      back_pts_.clear(); status_back_.clear(); err_back_.clear();
      cv::calcOpticalFlowPyrLK(curr_img_, prev_proc, s_next, back_pts_, status_back_, err_back_,
                               lk_win, lk_levels_,
                               cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-3),
                               0, 1e-4);
      p0_.reserve(s_prev.size()); p1_.reserve(s_prev.size());
      for (size_t i=0;i<s_next.size();++i) {
        if (!status_back_[i]) continue;
        const double fb = cv::norm(s_prev[i] - back_pts_[i]);
        if (fb < fb_max_error_px_) { p0_.push_back(s_prev[i]); p1_.push_back(s_next[i]); }
      }
    } else {
      p0_.swap(s_prev); p1_.swap(s_next);
    }

    // Robust trimming
    trim_outliers(p0_, p1_);
    if ((int)p0_.size() < 10) { initFeatures(curr_img_); resetFrameState(t_now); return; }

    // Estimate motion
    cv::Mat inliers;
    cv::Mat A = cv::estimateAffinePartial2D(
        p0_, p1_, inliers, cv::RANSAC,
        ransac_reproj_thresh_, 1000, 0.99, 10);

    double du_px = 0.0, dv_px = 0.0;
    int inl = 0;
    if (!A.empty()) {
      du_px = A.at<double>(0,2);
      dv_px = A.at<double>(1,2);
      inl = countInliersFlexible(inliers, (int)p0_.size());
    } else {
      du_px = robustMedianDelta(p0_, p1_, 0);
      dv_px = robustMedianDelta(p0_, p1_, 1);
      inl = (int)p0_.size();
    }

    const double shift_px = std::hypot(du_px, dv_px);
    const bool valid_meas = (inl >= min_inliers_publish_) && (shift_px <= max_shift_px_);

    // Convert to metric velocity (RAW)
    double vx_cam = -(du_px / fx_) * (Z / dt);
    double vy_cam =  (dv_px / fy_) * (Z / dt);
    if (compensate_rot_velocity_ && have_imu_) {
      vx_cam += Z * wy_;   // +h*q
      vy_cam -= Z * wx_;   // -h*p
    }

    // Clamp even valid velocities to avoid Z/dt jitter spikes
    vx_cam = clamp(vx_cam, -max_valid_v_mps_, max_valid_v_mps_);
    vy_cam = clamp(vy_cam, -max_valid_v_mps_, max_valid_v_mps_);

    if (!valid_meas) {
      RCLCPP_DEBUG_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Reject frame: inliers=%d (min=%d), shift=%.2f px (max=%.2f), dt=%.3f, Z=%.2f",
        inl, min_inliers_publish_, shift_px, max_shift_px_, dt, Z);
    }

    // Publish raw velocity (NaN on invalid so downstream can gate)
    geometry_msgs::msg::TwistStamped vmsg;
    vmsg.header.stamp = t_now; vmsg.header.frame_id = publish_frame_id_;
    vmsg.twist.linear.x = valid_meas ? vx_cam : std::numeric_limits<double>::quiet_NaN();
    vmsg.twist.linear.y = valid_meas ? vy_cam : std::numeric_limits<double>::quiet_NaN();
    vmsg.twist.linear.z = 0.0;
    vel_pub_->publish(vmsg);

    geometry_msgs::msg::TwistStamped pvmsg;
    pvmsg.header = vmsg.header;
    pvmsg.twist.linear.x = (valid_meas && Z>0.0) ? (vx_cam / Z) : std::numeric_limits<double>::quiet_NaN();
    pvmsg.twist.linear.y = (valid_meas && Z>0.0) ? (vy_cam / Z) : std::numeric_limits<double>::quiet_NaN();
    pvmsg.twist.linear.z = 0.0;
    pixelvel_pub_->publish(pvmsg);

    // --- Integrate position ---
    if (valid_meas || always_integrate_raw_) {
      double vx_int = vx_cam;
      double vy_int = vy_cam;
      if (!valid_meas) { // clamp raw when untrusted
        vx_int = clamp(vx_int, -raw_vmax_mps_, raw_vmax_mps_);
        vy_int = clamp(vy_int, -raw_vmax_mps_, raw_vmax_mps_);
      }
      x_cam_ += vx_int * dt;
      y_cam_ += vy_int * dt;
    }

    geometry_msgs::msg::PointStamped pmsg;
    pmsg.header = vmsg.header;
    pmsg.point.x = x_cam_;
    pmsg.point.y = y_cam_;
    pmsg.point.z = Z;
    position_pub_->publish(pmsg);

    // Prepare next / hygiene
    ++frame_count_;
    if (frame_count_ % reseed_every_ == 0) { initFeatures(curr_img_); shrinkScratch(); }
    else { refreshOrTrack(curr_img_, p1_, inliers); }
    resetFrameState(t_now);
  }

  // ------- Helpers -------
  inline rclcpp::Time now() { return this->get_clock()->now(); }

  inline double heightWithGrace() {
    if (have_valid_range_) {
      last_good_Z_ = height_m_; last_range_time_ = now(); return height_m_;
    }
    // grace window if range temporarily missing
    if ((now() - last_range_time_).seconds() <= range_grace_sec_ && std::isfinite(last_good_Z_)) {
      return last_good_Z_;
    }
    return height_fixed_;
  }

  void initFeatures(const cv::Mat &img) {
    prev_pts_.clear();
    prev_pts_.reserve(std::min(max_features_, 800));
    const int grid = 6;
    const int per_cell = std::max(3, max_features_ / (grid*grid));
    const int w = std::max(8, img.cols / grid);
    const int h = std::max(8, img.rows / grid);
    for (int gy=0; gy<grid; ++gy) {
      for (int gx=0; gx<grid; ++gx) {
        cv::Rect roi(gx*w, gy*h,
                     (gx==grid-1? img.cols - gx*w : w),
                     (gy==grid-1? img.rows - gy*h : h));
        cv::Rect roi2 = roi & cv::Rect(4,4,img.cols-8,img.rows-8);
        if (roi2.width <= 0 || roi2.height <= 0) continue;
        std::vector<cv::Point2f> cell;
        cv::goodFeaturesToTrack(img(roi2), cell, per_cell, quality_level_, min_distance_);
        for (auto &p : cell) p += cv::Point2f((float)roi2.x, (float)roi2.y);
        prev_pts_.insert(prev_pts_.end(), cell.begin(), cell.end());
      }
    }
  }

  void refreshOrTrack(const cv::Mat &img,
                      const std::vector<cv::Point2f> &p1,
                      const cv::Mat &inliers) {
    if (p1.empty() || inliers.empty()) { initFeatures(img); return; }
    p1_.clear(); p1_.reserve(p1.size());
    // Support both Nx1 and 1xN masks
    for (int i=0; i<inliers.rows; ++i) {
      const uchar ok = (inliers.cols == 1) ? inliers.at<uchar>(i,0) :
                        inliers.at<uchar>(0,i);
      if (ok) p1_.push_back(p1[(size_t)i]);
    }
    if ((int)p1_.size() < min_features_) initFeatures(img);
    else prev_pts_.swap(p1_);
  }

  inline void resetFrameState(const rclcpp::Time &t_now) {
    prev_time_ = t_now;
    curr_img_.copyTo(prev_img_);
  }

  void shrinkScratch() {
    std::vector<cv::Point2f>().swap(next_pts_);
    std::vector<cv::Point2f>().swap(back_pts_);
    std::vector<unsigned char>().swap(status_);
    std::vector<unsigned char>().swap(status_back_);
    std::vector<float>().swap(err_);
    std::vector<float>().swap(err_back_);
    std::vector<cv::Point2f>().swap(p0_);
    std::vector<cv::Point2f>().swap(p1_);
  }

  // ------- Members -------
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr caminfo_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr range_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr vel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pixelvel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr position_pub_;

  // camera intrinsics
  bool has_cam_info_ = false;
  double fx_full_=0, fy_full_=0, cx_full_=0, cy_full_=0;
  double fx_=0, fy_=0, cx_=0, cy_=0;
  cv::Size full_size_{0,0}, proc_size_{0,0};
  cv::Mat Kproc_, Kproc_inv_;
  bool have_maps_ = false;
  cv::Mat map1_, map2_;

  // sensors / range grace
  bool use_range_topic_ = true;
  bool have_valid_range_ = false;
  double height_m_ = std::numeric_limits<double>::quiet_NaN();
  double last_good_Z_ = std::numeric_limits<double>::quiet_NaN();
  rclcpp::Time last_range_time_;
  double height_fixed_ = 1.0;
  bool have_imu_ = false;
  double wx_=0, wy_=0, wz_=0;

  // state
  double x_cam_ = 0.0, y_cam_ = 0.0;

  // images/features
  std::vector<cv::Point2f> prev_pts_, next_pts_, back_pts_, p0_, p1_;
  std::vector<unsigned char> status_, status_back_;
  std::vector<float> err_, err_back_;
  cv::Mat gray_proc_, gray_proc_ud_, warped_prev_;
  cv::Mat curr_img_, prev_img_;
  rclcpp::Time prev_time_;

  // params
  std::string image_topic_, camera_info_topic_, range_topic_, imu_topic_, publish_frame_id_;
  int min_features_=120, max_features_=400;
  double quality_level_=0.01, min_distance_=8.0, ransac_reproj_thresh_=3.0, fb_max_error_px_=1.0;
  double proc_scale_=0.5;
  bool do_undistort_=false, do_clahe_=false, do_fb_check_=false, derotate_with_imu_=true, compensate_rot_velocity_=false;

  int lk_levels_=5, lk_win_px_=31;
  int min_inliers_publish_=15;
  double max_shift_px_=8.0;
  double max_dt_=0.5;
  double flow_eps_px_=0.10;

  // robustness helpers
  double range_grace_sec_=0.3;
  bool always_integrate_raw_=false;
  double raw_vmax_mps_=5.0;
  double max_valid_v_mps_=15.0;

  // hygiene
  size_t frame_count_ = 0;
  int reseed_every_ = 20;
  cv::Ptr<cv::CLAHE> clahe_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OpticalFlowVelocityNode>());
  rclcpp::shutdown();
  return 0;
}
