# ifndef __WORK_HPP__
# define __WORK_HPP__

# include <vector>
# include <iostream>
# include "trt_model.hpp"
# include "trt_logger.hpp"
# include "trt_classifierInit.hpp"
# include <opencv2/opencv.hpp>

namespace logger {
    enum class Level;
    class Logger;
}

namespace model {
    struct Params;
    namespace classifier {
        class Classifier;
    }
}

namespace worker{

class Worker
{
public:
    Worker(std::string onnxPath, logger::Level level, model::Params params);
    void inference(std::string imagePath);
    
    // 添加批处理图像的接口
    void inferenceWithBatch(const std::vector<cv::Mat>& images);

public:
    std::shared_ptr<logger::Logger>                 m_logger;
    std::shared_ptr<model::Params>                  m_params;
    std::shared_ptr<model::classifier::Classifier>  m_classifier;
    std::vector<float>                              m_scores;
};

// 工厂函数, 创建并返回一个Worker对象的智能指针(传入参数与构造函数相同)
std::shared_ptr<Worker> create_worker(std::string onnxPath, logger::Level level, model::Params params);
}; // namespace worker
# endif
