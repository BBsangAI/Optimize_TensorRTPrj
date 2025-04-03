# ifndef __TRT_MODEL_HPP__
# define __TRT_MODEL_HPP__

# include "trt_preprocess.hpp"

#define WORKSPACESIZE (128 * (1 << 20));

namespace model{
enum task_type{
    CLASSIFICATION,
    DETECTION,
    SEGMENTATION
};

enum device{
    CPU,
    GPU
};

enum precision{
    FP32,
    FP16,
    INT8
};
// 量化方法
enum calibrator{
    Entropy,
    MinMax
};

struct image_info{
    int t;
    int h;
    int w;
    int c;
    image_info(int channel, int temporal, int height, int widt) :c(channel), t(temporal), h(height), w(widt) {};
};

struct Params{
    task_type           task    = CLASSIFICATION;
    device              dev     = GPU;
    precision           prec    = FP16;
    calibrator          cal     = Entropy;
    process::tactics    tac     = process::tactics::GPU_BILINEAR;    //像素插值方法。（最近邻插值，双线性插值）
    image_info          img     = {3, 16, 112, 112};
    int                 ws_size = WORKSPACESIZE;
    int                 num_cls = 27;    
};

// 构建智能指针（自动释放）
template <typename T>
void destory_trt_ptr(T* ptr){
    if(ptr){
        std::string type_name = typeid(T).name();
        LOGD("destory %s", type_name.c_str());
        ptr->destroy();
    }
}

class Model{
public:
    Model(std::string onnx_path, logger::Level level, Params param);
    virtual ~Model() {};
    void load_image(std::string image_path);
    void init_model(); 
    void inference();

public:
    bool build_engine();
    bool load_engine();
    void save_plan(nvinfer1::IHostMemory& plan);
    void print_network(nvinfer1::INetworkDefinition &network, bool optimized);
    std::string getPrec(precision prec);

    bool enqueue_bindings();

    // 纯虚函数, 不同task的input与output不同, 自行实现
    // setup: 分配device与host端的memory, bindings, context
    virtual void setup(void const* data, std::size_t size) = 0;
    virtual bool preprocess_cpu()      = 0;
    virtual bool preprocess_gpu()      = 0;
    virtual bool postprocess_cpu()     = 0;
    virtual bool postprocess_gpu()     = 0;

    // 路径
    std::string m_imagePath;
    std::string m_onnxPath;
    std::string m_enginePath;

    // 参数
    Params* m_params;
    int     m_workspaceSize;
    float*  m_bindings[2];
    float*  m_inputMemory[2];
    float*  m_outputMemory[2];

    // shape
    nvinfer1::Dims m_inputDims;
    nvinfer1::Dims m_outputDims;
    cudaStream_t   m_stream;

    // 七大基本组件
    std::shared_ptr<logger::Logger>               m_logger;
    std::shared_ptr<timer::Timer>                 m_timer;
    std::shared_ptr<nvinfer1::IRuntime>           m_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine>        m_engine;
    std::shared_ptr<nvinfer1::IExecutionContext>  m_context;
    std::shared_ptr<nvinfer1::INetworkDefinition> m_network;
};
}


#endif