#include <QMainWindow>
#include "ui_myform.h"
#include "loadingUI.h"


// 定义MainWindow继承QMainWindow类
class MainWindow : public QMainWindow {
public:
    MainWindow(QWidget *parent = nullptr) : QMainWindow(parent), ui(new Ui::MainWindow){
        ui->setupUi(this); // 
        setWindowTitle("HDMI Output Example");
    }
private:
    Ui::MainWindow *ui;     // 已经设计好的Ui::MainWindow
};

// 定义LoadingUI类继承QMainWindow类
class LoadingWindow : public QMainWindow {
public:
    LoadingWindow(QWidget *parent = nullptr) : QMainWindow(parent), ui(new LoadingUi::MainWindow){
        ui->setupUi(this); // 
        setWindowTitle("LOADING...");
    }
private:
    LoadingUi::MainWindow *ui;     // 已经设计好的Ui::MainWindow
};