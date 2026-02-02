// --- START OF FILE text/plain ---
#include "camera.hpp"

#include <stdexcept>

#include "hikrobot/hikrobot.hpp"
#include "mindvision/mindvision.hpp"
#include "tools/yaml.hpp"
#include "tools/logger.hpp" // 引入 logger

namespace io
{
Camera::Camera(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto camera_name = tools::read<std::string>(yaml, "camera_name");
  
  // === 修改：读取焦段模式 ===
  std::string zoom_mode = "wide"; // 默认
  if (yaml["zoom_mode"]) {
      zoom_mode = tools::read<std::string>(yaml, "zoom_mode");
  } else {
      tools::logger()->warn("[Camera] 'zoom_mode' not found in config, using default 'wide'.");
  }

  // 确定参数前缀
  std::string prefix = (zoom_mode == "tele") ? "tele_" : "wide_";
  
  // 读取曝光 (优先读取带前缀的，没有则回退)
  double exposure_ms;
  if (yaml[prefix + "exposure_ms"]) {
      exposure_ms = tools::read<double>(yaml, prefix + "exposure_ms");
  } else {
      exposure_ms = tools::read<double>(yaml, "exposure_ms");
  }

  if (camera_name == "mindvision") {
    auto gamma = tools::read<double>(yaml, "gamma");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    camera_ = std::make_unique<MindVision>(exposure_ms, gamma, vid_pid);
  }
  else if (camera_name == "hikrobot") {
    // 读取增益 (优先读取带前缀的)
    double gain;
    if (yaml[prefix + "gain"]) {
        gain = tools::read<double>(yaml, prefix + "gain");
    } else {
        gain = tools::read<double>(yaml, "gain");
    }

    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    
    // 初始化相机
    camera_ = std::make_unique<HikRobot>(exposure_ms, gain, vid_pid);
    
    tools::logger()->info("[Camera] Initialized in '{}' mode. (Exp: {:.2f}ms, Gain: {:.1f})", zoom_mode, exposure_ms, gain);
  }
  else {
    throw std::runtime_error("Unknow camera_name: " + camera_name + "!");
  }
}

void Camera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  camera_->read(img, timestamp);
}

}  // namespace io