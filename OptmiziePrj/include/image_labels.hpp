#include <iostream>

#include <string>
#include <vector>
#include "assert.h"

using namespace std;

class ImageNetLabels {
public:
  ImageNetLabels()
  {
    for( int i=0 ; i<27; i++ )
    {
      string x = "NA";
      mLabels.push_back( x );
    }

    {
      mLabels[0]= "Doing_other_things";
      mLabels[1]= "Drumming_Fingers";
      mLabels[2]= "No_gesture";
      mLabels[3]= "Pulling_Hand_In";
      mLabels[4]= "Pulling_Two_Fingers_In";
      mLabels[5]= "Pushing_Hand_Away";
      mLabels[6]= "Pushing_Two_Fingers_Away";
      mLabels[7]= "Rolling_Hand_Backward";
      mLabels[8]= "Rolling_Hand_Forward";
      mLabels[9]= "Shaking_Hand";
      mLabels[10]= "Sliding_Two_Fingers_Down";
      mLabels[11]= "Sliding_Two_Fingers_Left";
      mLabels[12]= "Sliding_Two_Fingers_Right";
      mLabels[13]= "Sliding_Two_Fingers_Up";
      mLabels[14]= "Stop_Sign";
      mLabels[15]= "Swiping_Down";
      mLabels[16]= "Swiping_Left";
      mLabels[17]= "Swiping_Right";
      mLabels[18]= "Swiping_Up";
      mLabels[19]= "Thumb_Down";
      mLabels[20]= "Thumb_Up";
      mLabels[21]= "Turning_Hand_Clockwise";
      mLabels[22]= "Turning_Hand_Counterclockwise";
      mLabels[23]= "Zooming_In_With_Full_Hand";
      mLabels[24]= "Zooming_In_With_Two_Fingers";
      mLabels[25]= "Zooming_Out_With_Full_Hand";
      mLabels[26]= "Zooming_Out_With_Two_Fingers";
}
}

string imagenet_labelstring( int i ) {
  assert( i>=0 && i<27 );
  return mLabels[i];
}

private:
vector<string> mLabels;

};