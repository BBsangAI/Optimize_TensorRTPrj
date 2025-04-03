#ifndef MYFORM_H
#define MYFORM_H

#include <QMainWindow>
#include <QTimer>
#include <QImage>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "trt_work.hpp"

namespace Ui {
class MainWindow;
}

class MyForm : public QMainWindow
{
    Q_OBJECT

public:
    explicit MyForm(QWidget *parent = nullptr);
    ~MyForm();

    void setInferenceWorker(std::shared_ptr<worker::Worker> worker);
    void enqueueFrame(const cv::Mat& frame);

signals:
    void updateResult(const QString& result);


private slots:
    void on_pushButton_clicked();  // 开始检测
    void on_pushButton_2_clicked(); // 取消检测
    void updateFrame();

private:
    Ui::MainWindow *ui;
    QTimer *timer;
    QImage currentImage;
    cv::Mat currentFrame;
    std::mutex frameMutex;
    std::shared_ptr<worker::Worker> inferenceWorker;
    std::atomic<bool> isCapturing;

    void startCapture();
    void stopCapture();
};

#endif // MYFORM_H
