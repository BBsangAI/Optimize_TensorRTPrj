#include <iostream>

#include <string>
#include <vector>
#include "assert.h"

using namespace std;

class ImageNetLabels {
public:
  ImageNetLabels()
  {
    for( int i=0 ; i<1000; i++ )
    {
      string x = "NA";
      mLabels.push_back( x );
    }

    {
      mLabels[0]= "tench, Tinca tinca";
}
}

string imagenet_labelstring( int i ) {
  assert( i>=0 && i<1000 );
  return mLabels[i];
}

private:
vector<string> mLabels;

};