#ifndef __LOGGER_HPP__ // 防止头文件重复包含的保护宏
#define __LOGGER_HPP__

#include "NvInfer.h" // 包含TensorRT头文件
#include <string> // 包含字符串库
#include <stdarg.h> // 包含可变参数列表库
#include <memory> // 包含内存库，用于shared_ptr

// 定义不同级别的日志记录宏
#define LOGF(...) logger::Logger::__log_info(logger::Level::FATAL, __VA_ARGS__)
#define LOGE(...) logger::Logger::__log_info(logger::Level::ERROR, __VA_ARGS__)
#define LOGW(...) logger::Logger::__log_info(logger::Level::WARN,  __VA_ARGS__)
#define LOG(...)  logger::Logger::__log_info(logger::Level::INFO,  __VA_ARGS__)
#define LOGV(...) logger::Logger::__log_info(logger::Level::VERB,  __VA_ARGS__)
#define LOGD(...) logger::Logger::__log_info(logger::Level::DEBUG, __VA_ARGS__)
// 宏定义  (...) 可变参数数量         日志记录                         传给宏的剩余所有参数传给log_info函数
// 定义终端输出的颜色代码
#define DGREEN    "\033[1;36m"
#define BLUE      "\033[1;34m"
#define PURPLE    "\033[1;35m"
#define GREEN     "\033[1;32m"
#define YELLOW    "\033[1;33m"
#define RED       "\033[1;31m"
#define CLEAR     "\033[0m"

namespace logger {

// 日志级别枚举类
enum class Level : int32_t {
    FATAL = 0, // 致命错误级别
    ERROR = 1, // 错误级别
    WARN  = 2, // 警告级别
    INFO  = 3, // 信息级别
    VERB  = 4, // 详细信息级别
    DEBUG = 5  // 调试级别
};

// 继承自TensorRT ILogger的日志记录器类
class Logger : public nvinfer1::ILogger {

public:
    Logger(); // 默认构造函数
    Logger(Level level); // 带日志级别的构造函数
    virtual void log(Severity severity, const char* msg) noexcept override; // 重写日志记录函数
    static void __log_info(Level level, const char* format, ...); // 静态日志记录函数
    Severity get_severity(Level level); // 将日志级别转换为TensorRT严重程度
    Level get_level(Severity severity); // 将TensorRT严重程度转换为日志级别

private:
    static Level m_level; // 静态日志级别成员
    Severity m_severity; // TensorRT严重程度成员
};

// 创建日志记录器的工厂函数
std::shared_ptr<Logger> create_logger(Level level);

} // namespace logger

#endif //__LOGGER_HPP__
