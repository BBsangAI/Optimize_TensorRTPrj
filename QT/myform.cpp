#include "myform.h"
#include "ui_myform.h"
#include <QPixmap>
#include <QLabel>
#include <QVBoxLayout>

// 全局变量声明（将与main.cpp共享）
extern std::queue<cv::Mat> imgQueue;
extern std::mutex queueMutex;
extern std::condition_variable cv_not;
extern std::atomic<bool> isRunning; 
extern std::atomic<bool> isInference;

MyForm::MyForm(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    isCapturing(false)
{
    ui->setupUi(this);
    
    // 创建用于显示图像的QLabel
    QLabel *imgLabel = new QLabel(this);
    imgLabel->setObjectName("imageLabel");
    imgLabel->setGeometry(10, 10, 640, 380);
    imgLabel->setAlignment(Qt::AlignCenter);
    imgLabel->setScaledContents(true);
    
    // 创建定时器用于更新UI
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MyForm::updateFrame);
    timer->start(33); // ~30fps
}

MyForm::~MyForm()
{
    stopCapture();
    isRunning = false; // 停止线程
    delete ui;
    delete timer;
}


void MyForm::setInferenceWorker(std::shared_ptr<worker::Worker> worker)
{
    inferenceWorker = worker;
}

void MyForm::enqueueFrame(const cv::Mat& frame)
{
    std::lock_guard<std::mutex> lock(frameMutex);
    currentFrame = frame.clone();
}

void MyForm::on_pushButton_clicked()
{
    // 开始检测按钮
    startCapture();
    ui->textBrowser->setText("检测已启动");
}

void MyForm::on_pushButton_2_clicked()
{
    // 取消检测按钮
    stopCapture();
    ui->textBrowser->setText("检测已停止");
}

void MyForm::updateFrame()
{
    if (!isCapturing)
        return;

    QLabel *imgLabel = findChild<QLabel*>("imageLabel");
    if (!imgLabel)
        return;
    
    // 更新图像显示
    cv::Mat frameCopy;
    {
        std::lock_guard<std::mutex> lock(frameMutex);
        if (!currentFrame.empty()) {
            frameCopy = currentFrame.clone();
        }
    }
    
    if (!frameCopy.empty()) {
        cv::Mat rgbFrame;
        cv::cvtColor(frameCopy, rgbFrame, cv::COLOR_BGR2RGB);
        QImage qimg(rgbFrame.data, rgbFrame.cols, rgbFrame.rows, 
                    rgbFrame.step, QImage::Format_RGB888);
        imgLabel->setPixmap(QPixmap::fromImage(qimg));
    }
}

void MyForm::startCapture()
{
    isCapturing = true;
    isInference = true;
}

void MyForm::stopCapture()
{
    isCapturing = false;
    isInference = false;
   // cv_not.notify_all(); // 唤醒所有等待线程
}


