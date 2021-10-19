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

    // Called by OpenCV to check input for detectAndCompute(), but probably not needed
    // for PopSift
    /*
    if (image.empty() || image.depth() != CV_8U )
        CV_Error(Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");
    
    if (!mask.empty() && mask.type() != CV_8UC1)
        CV_Error(Error::StsBadArg, "mask has incorrect type (!=CV_8UC1");
    */

    uchar *img_ptr = image.ptr<uchar>(0); 
    std::unique_ptr<SiftJob> job(_popSift->enqueue(image.cols, image.rows, img_ptr));
    std::unique_ptr<popsift::Features> popFeatures(job->get());

    cout << "PopSIFT features count : " << popFeatures->getFeatureCount() << ", PopSIFT descriptor count :" << popFeatures->getDescriptorCount() << endl;

    for (const auto& popFeat: *popFeatures)
    {
        // Add every feature in the keypoints array
        KeyPoint kp = KeyPoint(popFeat.xpos, popFeat.ypos, popFeat.sigma);
        keypoints.push_back(kp);

        // Add every
        std::vector<unsigned char> descriptors(128);
        const popsift::Descriptor* popDesc = popFeat.desc[0];

        for (std::size_t i = 0; i < 128; i++) {
            descriptors.push_back(static_cast<unsigned char>(popDesc->features[i]));
        }




        // loop through all descriptors for this keypoint
        /*
        for (int orientationIndex = 0; orientationIndex < popFeat.num_ori; orientationIndex++)
        {} */


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
                OutputArray _descriptors)
{

    Mat image = _image.getMat();

    // Use popSift to detect keypoints/features and compute descriptors
    uchar *img_ptr = image.ptr<uchar>(0); 
    std::unique_ptr<SiftJob> job(_popSift->enqueue(image.cols, image.rows, img_ptr));
    std::unique_ptr<popsift::Features> popFeatures(job->get());

    cout << "PopSIFT features count : " << popFeatures->getFeatureCount() << ", PopSIFT descriptor count :" << popFeatures->getDescriptorCount() << endl;

    keypoints.reserve(popFeatures->getDescriptorCount());

    // magic input is copying SIFT input as implemented in openCV
    _descriptors.create((int)popFeatures->getDescriptorCount(), 128, CV_32F);
    Mat descriptors = _descriptors.getMat();

    cout << "POPSIFT::desc (rows, cols) : (" << descriptors.rows << ", " << descriptors.cols << ")" << endl;

    int matRow = 0;

    for (const auto& popFeat: *popFeatures)
    {
        // Add every feature in the keypoints array
        KeyPoint kp(popFeat.xpos, popFeat.ypos, popFeat.sigma, -1, 0, popFeat.debug_octave);

        for (int orientationIndex = 0; orientationIndex < popFeat.num_ori; ++orientationIndex) {
            
            keypoints.push_back(kp);

            const popsift:: Descriptor* popDesc = popFeat.desc[orientationIndex];

            float* row_ptr = descriptors.ptr<float>(matRow);
            for (std::size_t i = 0; i < 128; i++) {
                row_ptr[i] = popDesc->features[i];
            }

            ++matRow;
        }

        //const popsift::Descriptor* popDesc = popFeat.desc[0]; // just extracting one descriptor for now

    }

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
    config.setLevels(_params._numScales);
    config.setDownsampling(_params._firstOctave);
    config.setThreshold(_params._peakThreshold);
    config.setEdgeLimit(_params._edgeThreshold);
    config.setNormalizationMultiplier(9); // 2^9 = 512
    config.setNormMode(_params._rootSift ? popsift::Config::RootSift : popsift::Config::Classic);
    config.setDescMode(popsift::Config::VLFeat_Desc);

    // Used to limit to amount of keypoints found - not necessarily needed
    // config.setFilterMaxExtrema(_params._maxTotalKeypoints);
    // config.setFilterSorting(popsift::GridFilterConfig::LargestScaleFirst);
    
    _popSift.reset(new PopSift(config, popsift::Config::ExtractingMode, PopSift::ByteImages));

}