from PIL import Image
import cv2

def collect():
    cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=320,height=240,framerate=30/1 ! videoconvert ! videoscale ! video/x-raw,width=180,height=150 ! appsink", cv2.CAP_GSTREAMER)
    if not cap.isOpened():
            print("Error: Could not open video source.")
            return;
    i=0
    while(True):
        i += 1
        filename = f"../images_data/1/{i:05d}.jpg"
        ok, frame = cap.read()
        if not ok:
            print("Failed to capture image from camera.")
            return
        cv2.imwrite(filename, frame)
        print("save successful")
        cv2.waitKey (300)  
        cv2.destroyAllWindows()     

if __name__ == "__main__":
     collect()

