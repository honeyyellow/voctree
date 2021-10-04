// https://stackoverflow.com/questions/18348229/opencv-how-to-implement-featuredetector-interface

#include "PopSiftDetector.h"

using namespace cv;

PopSiftDetector::PopSiftDetector() {
    cout << "PopSiftDetector Constructorc called..." << endl;

}

Ptr<PopSiftDetector> PopSiftDetector::create() {
    cout << "PopSift create() called..." << endl;
    return makePtr<PopSiftDetector>();
}

void PopSiftDetector::detect(InputArray image,
                CV_OUT std::vector<KeyPoint> &keypoints,
                InputArray mask)
{
    cout << "PopSift detect() called..." << endl;
}

void PopSiftDetector::detect(InputArrayOfArrays images,
                CV_OUT std::vector<std::vector<KeyPoint> > &keypoints,
                InputArrayOfArrays masks)
{
    cout << "PopSift detect() called..." << endl;
}

void PopSiftDetector::compute(InputArray image,
                CV_OUT CV_IN_OUT std::vector<KeyPoint> &keypoints,
                OutputArray descriptors)
{
    cout << "PopSift compute() called..." << endl;

}

void PopSiftDetector::compute(InputArrayOfArrays image,
                CV_OUT CV_IN_OUT std::vector<std::vector<KeyPoint> > &keypoints,
                OutputArrayOfArrays descriptors )
{
    cout << "PopSift compute() called..." << endl;
}