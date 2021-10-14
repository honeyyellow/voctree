// https://stackoverflow.com/questions/18348229/opencv-how-to-implement-featuredetector-interface

#include "PopSiftDetector.h"

using namespace cv;

std::unique_ptr<PopSift> PopSiftDetector::_popSift{nullptr};

PopSiftDetector::PopSiftDetector() {
    cout << "PopSiftDetector Constructor called..." << endl;
    if (_popSift == nullptr)
        resetConfiguration();

}

Ptr<PopSiftDetector> PopSiftDetector::create() {
    cout << "PopSiftDetector::create() called..." << endl;
    return makePtr<PopSiftDetector>();
}

void PopSiftDetector::detect(InputArray _image,
                CV_OUT std::vector<KeyPoint> &keypoints,
                InputArray mask)
{
    cout << "PopSiftDetector::detect() called..." << endl;
    //TODO - call popsift in order to detect and compute 

    Mat image = _image.getMat();

    uchar *img_ptr = image.ptr<uchar>(0); // This is supposed to work given that 
    std::unique_ptr<SiftJob> job(_popSift->enqueue(image.cols, image.rows, img_ptr));
    std::unique_ptr<popsift::Features> popFeatures(job->get());

    cout << "PopSIFT features count : " << popFeatures->getFeatureCount() << ", PopSIFT descriptor count :" << popFeatures->getDescriptorCount() << endl;

    for (const auto& popFeat: *popFeatures)
    {
        // Add every feature in the keypoints array
        // What kind of type is popFeat?

        for (int orientationIndex = 0; orientationIndex < popFeat.num_ori; orientationIndex++)
        {
            
        }
    }


}

void PopSiftDetector::detect(InputArrayOfArrays images,
                CV_OUT std::vector<std::vector<KeyPoint> > &keypoints,
                InputArrayOfArrays masks)
{
    cout << "PopSiftDetector::detect() called..." << endl;
}

void PopSiftDetector::compute(InputArray _image,
                CV_OUT CV_IN_OUT std::vector<KeyPoint> &keypoints,
                OutputArray descriptors)
{
    cout << "PopSiftDetector::compute() called..." << endl;

}

void PopSiftDetector::compute(InputArrayOfArrays image,
                CV_OUT CV_IN_OUT std::vector<std::vector<KeyPoint> > &keypoints,
                OutputArrayOfArrays descriptors )
{
    cout << "PopSiftDetector::compute() called..." << endl;
}

int PopSiftDetector::defaultNorm() const
{
    return NORM_L2;
}

void PopSiftDetector::resetConfiguration()
{
    // destroy all allocations and reset all state
    // on the current device in the current process
    cudaDeviceReset();

    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set(0, true);

    popsift::Config config;
    // Ask Carsten about these values, do I create a SiftParams, or do I 
    // reset configuration
    config.setLevels(_params._numScales);
    config.setDownsampling(_params._firstOctave);
    config.setThreshold(_params._peakThreshold);
    config.setEdgeLimit(_params._edgeThreshold);
    config.setNormalizationMultiplier(9); // 2^9 = 512
    config.setNormMode(_params._rootSift ? popsift::Config::RootSift : popsift::Config::Classic);
    config.setFilterMaxExtrema(_params._maxTotalKeypoints);
    config.setFilterSorting(popsift::GridFilterConfig::LargestScaleFirst);
    
    _popSift.reset(new PopSift(config, popsift::Config::ExtractingMode, PopSift::ByteImages));




}