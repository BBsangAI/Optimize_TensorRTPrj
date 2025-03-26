
//gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoscale ! videoconvert ! video/x-raw,width=180,height=150 ! autovideosink^C


//gst-launch-1.0 v4l2src device=/dev/video0 ! "image/jpeg,width=800,height=600,framerate=60/1" ! jpegdec ! videoconvert ! videoscale! video/x-raw,width=180,height=150 ! ximagesink



