#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include <camera_info_manager/camera_info_manager.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

namespace
{

std::string strip_leading_slashes(std::string value)
{
  while (!value.empty() && value.front() == '/') {
    value.erase(value.begin());
  }
  return value.empty() ? "camera_optical_frame" : value;
}

std::string file_url(const std::string & path)
{
  if (path.rfind("file://", 0) == 0 || path.empty()) {
    return path;
  }
  const auto generic = std::filesystem::absolute(std::filesystem::path(path)).generic_string();
#ifdef _WIN32
  return "file:///" + generic;
#else
  return "file://" + generic;
#endif
}

rclcpp::QoS image_qos(const std::string & reliability)
{
  auto qos = rclcpp::QoS(rclcpp::KeepLast(5)).durability_volatile();
  if (reliability == "best_effort") {
    qos.best_effort();
  } else if (reliability != "reliable") {
    throw std::invalid_argument("qos_reliability must be 'reliable' or 'best_effort'");
  } else {
    qos.reliable();
  }
  return qos;
}

sensor_msgs::msg::CameraInfo default_camera_info(
  int width, int height, const std::string & camera_name, const std::string & frame_id)
{
  sensor_msgs::msg::CameraInfo info;
  info.header.frame_id = frame_id;
  info.width = static_cast<uint32_t>(std::max(width, 0));
  info.height = static_cast<uint32_t>(std::max(height, 0));
  info.distortion_model = "plumb_bob";
  info.d.assign(5, 0.0);

  const auto fx = width > 0 ? static_cast<double>(width) : 1.0;
  const auto fy = height > 0 ? static_cast<double>(height) : 1.0;
  const auto cx = width > 0 ? static_cast<double>(width) / 2.0 : 0.0;
  const auto cy = height > 0 ? static_cast<double>(height) / 2.0 : 0.0;
  info.k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
  info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  info.p = {fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0};
  (void)camera_name;
  return info;
}

}  // namespace

class OpenCVCameraNode : public rclcpp::Node
{
public:
  OpenCVCameraNode()
  : Node("opencv_camera")
  {
    const auto camera_index = declare_parameter<int>("camera_index", 0);
    const auto video_file = declare_parameter<std::string>("video_file", "");
    const auto calibration_file = declare_parameter<std::string>("calibration_file", "");
    const auto camera_name = declare_parameter<std::string>("camera_name", "camera");
    frame_id_ = strip_leading_slashes(declare_parameter<std::string>("frame_id", "camera_optical_frame"));
    const auto width = declare_parameter<int>("width", 640);
    const auto height = declare_parameter<int>("height", 480);
    const auto fps = declare_parameter<double>("fps", 30.0);
    loop_ = declare_parameter<bool>("loop", false);
    target_width_ = width;
    target_height_ = height;
    jpeg_quality_ = static_cast<int>(std::clamp<int64_t>(declare_parameter<int>("jpeg_quality", 80), 1, 100));
    const auto qos = image_qos(declare_parameter<std::string>("qos_reliability", "reliable"));

    if (video_file.empty()) {
      capture_.open(camera_index);
      source_name_ = std::to_string(camera_index);
    } else {
      capture_.open(video_file);
      source_name_ = video_file;
    }
    if (!capture_.isOpened()) {
      throw std::runtime_error("Could not open camera source: " + source_name_);
    }

    if (width > 0) {
      capture_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    }
    if (height > 0) {
      capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }
    if (fps > 0.0) {
      capture_.set(cv::CAP_PROP_FPS, fps);
    }

    const auto actual_width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    const auto actual_height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    const auto info_width = target_width_ > 0 ? target_width_ : actual_width;
    const auto info_height = target_height_ > 0 ? target_height_ : actual_height;
    camera_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name);
    if (!calibration_file.empty() && !camera_info_manager_->loadCameraInfo(file_url(calibration_file))) {
      RCLCPP_WARN(get_logger(), "Could not load calibration file '%s'", calibration_file.c_str());
    }
    camera_info_ = camera_info_manager_->isCalibrated() ?
      camera_info_manager_->getCameraInfo() :
      default_camera_info(info_width, info_height, camera_name, frame_id_);
    camera_info_.header.frame_id = frame_id_;
    if (camera_info_.width == 0) {
      camera_info_.width = static_cast<uint32_t>(std::max(info_width, 0));
    }
    if (camera_info_.height == 0) {
      camera_info_.height = static_cast<uint32_t>(std::max(info_height, 0));
    }

    image_pub_ = create_publisher<sensor_msgs::msg::Image>("image_raw", qos);
    compressed_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>("image_raw/compressed", qos);
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", qos);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / std::max(fps, 1.0))),
      std::bind(&OpenCVCameraNode::publish_frame, this));

    RCLCPP_INFO(
      get_logger(), "Publishing %s as image_raw/camera_info at %.2f FPS",
      source_name_.c_str(), std::max(fps, 1.0));
  }

private:
  void publish_frame()
  {
    cv::Mat frame;
    if (!capture_.read(frame) && loop_) {
      capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
      capture_.read(frame);
    }
    if (frame.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "No frame available from %s", source_name_.c_str());
      return;
    }
    const auto captured_size = frame.size();
    if (target_width_ > 0 && target_height_ > 0 &&
      (frame.cols != target_width_ || frame.rows != target_height_))
    {
      cv::resize(frame, frame, cv::Size(target_width_, target_height_), 0.0, 0.0, cv::INTER_AREA);
    }
    if (!logged_frame_format_) {
      logged_frame_format_ = true;
      RCLCPP_INFO(
        get_logger(), "Captured %dx%d, publishing %dx%d",
        captured_size.width, captured_size.height, frame.cols, frame.rows);
    }

    auto header = std_msgs::msg::Header();
    header.stamp = now();
    header.frame_id = frame_id_;
    auto image = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, frame).toImageMsg();
    auto info = camera_info_;
    info.header = header;
    image_pub_->publish(*image);
    info_pub_->publish(info);

    if (compressed_pub_->get_subscription_count() > 0) {
      std::vector<unsigned char> encoded;
      cv::imencode(".jpg", frame, encoded, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_});
      sensor_msgs::msg::CompressedImage compressed;
      compressed.header = header;
      compressed.format = "jpeg";
      compressed.data.assign(encoded.begin(), encoded.end());
      compressed_pub_->publish(std::move(compressed));
    }
  }

  cv::VideoCapture capture_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  sensor_msgs::msg::CameraInfo camera_info_;
  std::string frame_id_;
  std::string source_name_;
  int target_width_ = 640;
  int target_height_ = 480;
  int jpeg_quality_ = 80;
  bool loop_ = false;
  bool logged_frame_format_ = false;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<OpenCVCameraNode>());
  } catch (const std::exception & exc) {
    RCLCPP_FATAL(rclcpp::get_logger("opencv_camera"), "%s", exc.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
