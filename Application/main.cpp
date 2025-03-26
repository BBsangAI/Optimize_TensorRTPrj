# include <string>
# include <iostream>
# include <thread>
# include <mutex>
# include <queue>
# include <condition_variable>
# include <atomic>
# include <opencv2/opencv.hpp>
# include "trt_model.hpp"
# include "trt_work.hpp"
# include "trt_logger.hpp"

// 图像队列及同步机制
std::queue<cv::Mat> imgQueue;
std::mutex queueMutex;
std::condition_variable cv;
std::atomic<bool> isRunning(true);
const int BATCH_SIZE = 16;

// 图像采集线程函数
void captureThread() {
    std::string pipeline = "v4l2src device=/dev/video0 ! video/x-raw,format=YUY2, "
                           "width=320,height=240,framerate=30/1 ! videoconvert ! videoscale ! "
                           "video/x-raw,width=180,height=150 ! appsink";
    // 打开摄像头
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头！" << std::endl;
        isRunning = false;
        return;
    }

    while (isRunning) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "无法读取摄像头图像！" << std::endl;
            break;
        }

        // 将图像放入队列
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            imgQueue.push(frame.clone());
            
            // 队列容量控制，避免内存溢出
            if (imgQueue.size() > BATCH_SIZE * 3) {
                imgQueue.pop();
            }
        }
        cv.notify_one(); // 通知推理线程

        // 控制采集速度
        std::this_thread::sleep_for(std::chrono::milliseconds(30)); // ~30fps
    }
    
    cap.release();
}

// 推理线程函数
void inferenceThread(std::shared_ptr<worker::Worker> worker) {
    std::vector<cv::Mat> batch;
    batch.reserve(BATCH_SIZE);
    
    while (isRunning) {
        // 等待至少有一帧图像可用
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            cv.wait(lock, []{ return !imgQueue.empty() || !isRunning; });
            
            if (!isRunning && imgQueue.empty()) break;
            
            // 收集批次
            while (!imgQueue.empty() && batch.size() < BATCH_SIZE) {
                batch.push_back(imgQueue.front());
                imgQueue.pop();
            }
        }
        
        // 处理批次
        if (!batch.empty()) {
            // 如果不足16帧，复制最后一帧填充
            while (batch.size() < BATCH_SIZE) {
                if (!batch.empty()) {
                    batch.push_back(batch.back().clone());
                } else {
                    break;
                }
            }
            
            if (batch.size() == BATCH_SIZE) {
                worker->inferenceWithBatch(batch);
                std::cout << "完成一批次(16帧)推理" << std::endl;
            }
            
            batch.clear();
        }
    }
}

int main(){
    std::string onnxpath = "../model/mobilenetv2.onnx";
    auto level = logger::Level::VERB;
    auto params = model::Params();

    params.img = {3, 16, 112, 112}; // 批次大小为16
    params.num_cls = 27;
    params.task = model::task_type::CLASSIFICATION;
    params.dev = model::device::GPU; 
    params.prec = model::precision::FP16;
    params.cal = model::calibrator::Entropy;

    auto worker = worker::create_worker(onnxpath, level, params);
    
    // 启动线程
    std::thread capture(captureThread);
    std::thread inference(inferenceThread, worker);
    
    // 等待用户输入退出
    std::cout << "按任意键退出程序..." << std::endl;
    std::cin.get();
    
    // 停止线程
    isRunning = false;
    cv.notify_all();
    
    // 等待线程结束
    capture.join();
    inference.join();
    
    std::cout << "程序已退出" << std::endl;
    return 0;
}