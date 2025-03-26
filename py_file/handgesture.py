import cv2
import mediapipe as mp

# 初始化 MediaPipe 手部检测模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 初始化手部检测器
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 打开视频输入（0 表示默认摄像头）
cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=320,height=240,framerate=30/1 ! videoconvert ! videoscale ! video/x-raw,width=180,height=150 ! appsink", cv2.CAP_GSTREAMER)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # 将图像转换为 RGB 格式，因为 MediaPipe 使用 RGB 图像
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 进行手部检测
    results = hands.process(image_rgb)

    # 绘制检测结果
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # 绘制每只手的关键点
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
    
    # 显示处理后的帧
    cv2.imshow("Hand Gesture Detection", frame)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕捉对象并关闭窗口
cap.release()
cv2.destroyAllWindows()