/**
 * Created 3. Oct 2021
 * 
 */

#ifndef POPSIFTDETECTOR_H_
#define POPSIFTDETECTOR_H_

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <fstream>

#include <opencv2/features2d/features2d.hpp>

// PopSIFT includes
#include <popsift/popsift.h>
#include <popsift/sift_pyramid.h>
#include <popsift/sift_octave.h>
#include <popsift/common/device_prop.h>

#include "SiftParams.h"

using namespace std;

//class PopSift;

namespace cv {

class PopSiftDetector: public Feature2D {
public:

    PopSiftDetector();

    static Ptr<PopSiftDetector> create();

    CV_WRAP void detect(InputArray _image,
                CV_OUT std::vector<KeyPoint> &keypoints,
                InputArray mask = noArray() );
    
    CV_WRAP void detect(InputArrayOfArrays images,
                CV_OUT std::vector<std::vector<KeyPoint> > &keypoints,
                InputArrayOfArrays masks = noArray() );
    
    CV_WRAP void compute(InputArray _image,
                CV_OUT CV_IN_OUT std::vector<KeyPoint> &keypoints,
                OutputArray _descriptors );
    
    CV_WRAP void compute(InputArrayOfArrays images,
                CV_OUT CV_IN_OUT std::vector<std::vector<KeyPoint> > &keypoints,
                OutputArrayOfArrays descriptors );
    
    int defaultNorm() const CV_OVERRIDE;

private:

    SiftParams _params; //TODO - what to do with this?? Check SiftParams.h
    bool _isOriented = true;
    static std::unique_ptr<PopSift> _popSift;
    void resetConfiguration();
};

} /* namespace cv */

#endif /* POPSIFTDETECTOR_H_ */

