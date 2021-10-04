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

using namespace std;

namespace cv {

class PopSiftDetector: public FeatureDetector {
public:

    PopSiftDetector();

    static Ptr<PopSiftDetector> create();

    CV_WRAP void detect(InputArray image,
                CV_OUT std::vector<KeyPoint> &keypoints,
                InputArray mask = noArray() );
    
    CV_WRAP void detect(InputArrayOfArrays images,
                CV_OUT std::vector<std::vector<KeyPoint> > &keypoints,
                InputArrayOfArrays masks = noArray() );
    
    CV_WRAP void compute(InputArray image,
                CV_OUT CV_IN_OUT std::vector<KeyPoint> &keypoints,
                OutputArray descriptors );
    
    CV_WRAP void compute(InputArrayOfArrays image,
                CV_OUT CV_IN_OUT std::vector<std::vector<KeyPoint> > &keypoints,
                OutputArrayOfArrays descriptors );
    
};

} /* namespace cv */

#endif /* POPSIFTDETECTOR_H_ */

