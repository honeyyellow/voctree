
#ifndef _SIFTPARAMS_H_
#define _SIFTPARAMS_H_

#include <stdlib.h>

using namespace std;

// Ask Carsten about this struct and differences from the
// struct defined here: https://github.com/alicevision/AliceVision/blob/develop/src/aliceVision/feature/sift/SIFT.hpp
struct SiftParams
{
    /// Use original image, or perform an upscale if == -1
  int _firstOctave = 0;
  /// Scales per octave
  int _numScales = 3;
  /// Max ratio of Hessian eigenvalues
  float _edgeThreshold = 10.0f;
  /// Min contrast
  float _peakThreshold = 0.005f;
  /// Min contrast (relative to variance median)
  float _relativePeakThreshold = 0.01f;

  size_t _gridSize = 4;
  size_t _maxTotalKeypoints = 10000;
  /// see [1]
  bool _rootSift = false;

  int getImageFirstOctave(int w, int h) const
  {
    return _firstOctave - (w * h <= 3000 * 2000 ? 1 : 0); // -1 to upscale for small resolutions
  }
};

#endif