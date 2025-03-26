#include "trt_logger.hpp" // 包含日志记录器的头文件
#include "NvInfer.h" // 包含TensorRT头文件
#include <cstdlib> // 包含标准库，用于exit函数

using namespace std; // 使用标准命名空间

namespace logger { // 开始logger命名空间

Level Logger::m_level = Level::INFO; // 初始化静态成员m_level为INFO级别

Logger::Logger(Level level) { // 使用指定日志级别初始化记录器的构造函数
    m_level = level; // 设置日志级别
    m_severity = get_severity(level); // 获取对应的TensorRT严重程度
}

Logger::Severity Logger::get_severity(Level level) { // 将日志级别转换为TensorRT的严重程度
    switch (level) {
        case Level::FATAL: return Severity::kINTERNAL_ERROR; // 致命错误映射到内部错误
        case Level::ERROR: return Severity::kERROR; // 错误级别映射
        case Level::WARN:  return Severity::kWARNING; // 警告级别映射
        case Level::INFO:  return Severity::kINFO; // 信息级别映射
        case Level::VERB:  return Severity::kVERBOSE; // 详细信息级别映射
        default:           return Severity::kVERBOSE; // 默认为详细信息级别
    }
}

Level Logger::get_level(Severity severity) { // 将TensorRT的严重程度转换为日志级别
    string str;
    switch (severity) {
        case Severity::kINTERNAL_ERROR: return Level::FATAL; // 内部错误映射到致命错误
        case Severity::kERROR:          return Level::ERROR; // 错误映射到错误级别
        case Severity::kWARNING:        return Level::WARN; // 警告映射到警告级别
        case Severity::kINFO:           return Level::INFO; // 信息映射到信息级别
        case Severity::kVERBOSE:        return Level::VERB; // 详细信息映射到详细信息级别
    }
}

void Logger::log (Severity severity, const char* msg) noexcept { // 记录指定严重程度的日志消息
    // 仅当严重程度不高于WARNING或日志级别为DEBUG时才记录
    if (severity <= get_severity(Level::WARN) || m_level >= Level::DEBUG)
        __log_info(get_level(severity), "%s", msg); // 记录消息
}

void Logger::__log_info(Level level, const char* format, ...) { // 内部日志记录函数
    char msg[1000]; // 消息缓冲区
    va_list args; // 可变参数列表
    va_start(args, format); // 初始化可变参数列表
    int n = 0; // 消息长度
    
    // 根据日志级别添加相应的彩色前缀
    switch (level) {
        case Level::DEBUG: n += snprintf(msg + n, sizeof(msg) - n, DGREEN "[调试]" CLEAR); break;
        case Level::VERB:  n += snprintf(msg + n, sizeof(msg) - n, PURPLE "[详细]" CLEAR); break;
        case Level::INFO:  n += snprintf(msg + n, sizeof(msg) - n, YELLOW "[信息]" CLEAR); break;
        case Level::WARN:  n += snprintf(msg + n, sizeof(msg) - n, BLUE "[警告]" CLEAR); break;
        case Level::ERROR: n += snprintf(msg + n, sizeof(msg) - n, RED "[错误]" CLEAR); break;
        default:           n += snprintf(msg + n, sizeof(msg) - n, RED "[致命]" CLEAR); break;
    }

    n += vsnprintf(msg + n, sizeof(msg) - n, format, args); // 格式化消息

    va_end(args); // 清理可变参数列表

    if (level <= m_level) // 如果消息级别低于或等于记录器级别
        fprintf(stdout, "%s\n", msg); // 将消息打印到标准输出

    if (level <= Level::ERROR) { // 如果是错误或致命错误级别
        fflush(stdout); // 刷新标准输出
        exit(0); // 退出程序
    }
}

shared_ptr<Logger> create_logger(Level level) { // 工厂函数创建记录器
    return make_shared<Logger>(level); // 返回指向新记录器的共享指针
}

} // namespace logger
