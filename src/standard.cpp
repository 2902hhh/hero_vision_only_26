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
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

using namespace std::chrono;

// 1. 修改命令行参数，增加 display 选项，默认开启 (true)
const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{display d      | true | 是否显示视频流}" 
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  
  // 获取是否显示的标志
  bool enable_display = cli.get<bool>("display");

  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  
  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  auto mode = io::GimbalMode::IDLE;
  auto last_mode = io::GimbalMode::IDLE;

  // 用于计算FPS
  int frame_count = 0;
  auto last_fps_time = std::chrono::steady_clock::now();
  double fps = 0.0;

  while (!exiter.exit()) {
    // 简单计算FPS
    frame_count++;
    auto now = std::chrono::steady_clock::now();
    if (tools::delta_time(now, last_fps_time) >= 1.0) {
        fps = frame_count / tools::delta_time(now, last_fps_time);
        frame_count = 0;
        last_fps_time = now;
    }

    camera.read(img, t);
    // 增加空图检查，防止程序崩溃
    if (img.empty()) continue;

    auto gs = gimbal.state(); 
    q = gimbal.q(t - 1ms);
    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode;
    }

    // recorder.record(img, q, t);

    solver.set_R_gimbal2world(q);

    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    // 1. 识别
    auto armors = detector.detect(img);

    // 2. 追踪
    auto targets = tracker.track(armors, t);

    // 3. 瞄准
    auto command = aimer.aim(targets, t, gs.bullet_speed);

    command.shoot = shooter.shoot(command, aimer, targets, ypr); 
    
    gimbal.send(command);

    // ==================== 可视化代码开始 ====================
    if (enable_display) {
        // 克隆一份图像用于绘制，避免影响原图处理
        cv::Mat vis_img = img.clone();

        // A. 绘制识别到的装甲板 (绿色框)
        for (const auto & armor : armors) {
            // 绘制四点连线
            for (int i = 0; i < 4; i++) {
                cv::line(vis_img, armor.points[i], armor.points[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);
            }
            // 绘制中心点
            cv::circle(vis_img, armor.center, 3, cv::Scalar(0, 255, 0), -1);
            // 显示装甲板名称
            tools::draw_text(vis_img, auto_aim::ARMOR_NAMES[armor.name], armor.points[0], cv::Scalar(0, 255, 0), 0.8, 2);
        }

        // B. 绘制追踪预测结果 (黄色框)
        if (!targets.empty()) {
            auto target = targets.front();
            
            // 获取目标所有装甲板的预测位置 (世界坐标系)
            std::vector<Eigen::Vector4d> predicted_armors = target.armor_xyza_list();
            
            for (const auto & xyza : predicted_armors) {
                // 关键步骤：重投影 (Reprojection)
                // 利用 PnP 的逆过程，将预测的世界坐标(3D) 映射回 图像像素坐标(2D)
                auto image_points = solver.reproject_armor(
                    xyza.head(3), xyza[3], target.armor_type, target.name
                );
                
                // 绘制预测框
                tools::draw_points(vis_img, image_points, {0, 255, 255}, 2);

                // // 2. 计算距离 (欧氏距离: sqrt(x^2 + y^2 + z^2))
                // double distance = xyza.head(3).norm();

                // // 3. 在装甲板上方显示距离
                // // image_points[0] 通常是装甲板的左上角或某个角点，适合作为文本锚点
                // if (!image_points.empty()) {
                //     std::string dist_text = fmt::format("{:.2f}m", distance);
                //     // 稍微向上偏移一点 (y - 20) 以免挡住框
                //     cv::Point text_pos = image_points[0];
                //     text_pos.y -= 20; 
                    
                //     tools::draw_text(vis_img, dist_text, text_pos, {0, 255, 255}, 0.6, 2);
                // }
            }

            // 在左上角显示追踪状态
            std::string state_info = fmt::format("State: {} | ID: {}", tracker.state(), auto_aim::ARMOR_NAMES[target.name]);
            tools::draw_text(vis_img, state_info, {20, 80}, {0, 255, 255}, 1.0, 2);
        }

        // C. 绘制最终打击点 (红色十字)
        // aimer.debug_aim_point 存储了经过反小陀螺/前哨站策略选择后的最终击打位置
        if (aimer.debug_aim_point.valid) {
            Eigen::Vector3d aim_xyz = aimer.debug_aim_point.xyza.head(3);
            
            // 同样使用重投影将其画在图上
            // 这里的参数 0, small, outpost 只是为了借用函数接口，投影中心点
            auto aim_proj_points = solver.reproject_armor(aim_xyz, 0, auto_aim::ArmorType::small, auto_aim::ArmorName::outpost);
            
            // 计算投影矩形的中心
            cv::Point2f aim_center = (aim_proj_points[0] + aim_proj_points[2]) / 2;
            
            // 画红十字
            cv::line(vis_img, {aim_center.x - 15, aim_center.y}, {aim_center.x + 15, aim_center.y}, {0, 0, 255}, 2);
            cv::line(vis_img, {aim_center.x, aim_center.y - 15}, {aim_center.x, aim_center.y + 15}, {0, 0, 255}, 2);
        }

        // D. 绘制 UI 信息
        tools::draw_text(vis_img, fmt::format("FPS: {:.1f}", fps), {20, 40}, {255, 255, 255}, 1.0, 2);
        tools::draw_text(vis_img, fmt::format("Mode: {}", gimbal.str(mode)), {20, 120}, {255, 255, 255}, 1.0, 2);

        // E. 显示图像 (缩小一半显示，防止超出屏幕)
        cv::resize(vis_img, vis_img, {}, 0.5, 0.5);
        cv::imshow("Auto Aim Debug", vis_img);
        
        // 必须加 waitKey，否则窗口不刷新
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    // ==================== 可视化代码结束 ====================
  }

  return 0;
}