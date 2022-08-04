#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <stdlib.h>
#include <assert.h>
#include <vector>

#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREAD_PER_BLOCK 127

#define PI 3.1415926535897932384626433832795

// BASIC STRUCTURES:

#define IMAGE_WIDTH             (486)  /* input image width */
#define IMAGE_HEIGHT            (362)  /* input image height */

// Keypoints:
#define OriSize1  8
#define IndexSize1  4

#define OriSize     8
#define IndexSize   4

const int VecLength = IndexSize * IndexSize * OriSize;
typedef std::vector<float> floatVect;

#define MIN(a,b) (a)<(b)?(a):(b)

//#define ACCURACY 0.0000001
//#define ACCURACY 0.000001
#define ACCURACY 1

/* Keypoint structure:
	position:   x,y
	scale:      s
	orientation:    angle
	descriptor: array of gradient orientation histograms in a neighbors */
struct keypoint
{
	float   x, y,
		scale, radius,
		angle;

	float   vec[IndexSize * IndexSize * OriSize];
};

/* List of keypoints: just use the standard class vector: */
typedef std::vector<keypoint> keypointslist;

struct keypointHolder
{
	int imWidth;
	int imHeight;
	int imOffset;
	float octSize;
	float octScale;
	float octRow;
	float octCol;
	float angle;
};

/* List of keypoints: just use the standard class vector: */
typedef std::vector<keypointHolder> keypointHolderVect;


struct siftPar
{
	int totalCount;

	int OctaveMax;

	int DoubleImSize;

	int order;


	/* InitSigma gives the amount of smoothing applied to the image at the
	   first level of each octave.  In effect, this determines the sampling
	   needed in the image domain relative to amount of smoothing.  Good
	   values determined experimentally are in the range 1.2 to 1.8.
	*/
	float  InitSigma /*= 1.6*/;


	/* Peaks in the DOG function must be at least BorderDist samples away
	   from the image border, at whatever sampling is used for that scale.
	   Keypoints close to the border (BorderDist < about 15) will have part
	   of the descriptor landing outside the image, which is approximated by
	   having the closest image pixel replicated.  However, to perform as much
	   matching as possible close to the edge, use BorderDist of 4.
	*/
	int BorderDist /*= 5*/;


	/* Scales gives the number of discrete smoothing levels within each octave.
	   For example, Scales = 2 implies dividing octave into 2 intervals, so
	   smoothing for each scale sample is sqrt(2) more than previous level.
	   Value of 2 works well, but higher values find somewhat more keypoints.
	*/

	int Scales /*= 3*/;


	/// Decreasing PeakThresh allows more non contrasted keypoints
	/* Magnitude of difference-of-Gaussian value at a keypoint must be above
	   this threshold.  This avoids considering points with very low contrast
	   that are dominated by noise.  It is divided by Scales because more
	   closely spaced scale samples produce smaller DOG values.  A value of
	   0.08 considers only the most stable keypoints, but applications may
	   wish to use lower values such as 0.02 to find keypoints from low-contast
	   regions.
	*/

	//#define  PeakThreshInit  255*0.04 
	//#define  PeakThresh      PeakThreshInit / Scales
	float PeakThresh  /*255.0 * 0.04 / 3.0*/;

	/// Decreasing EdgeThresh allows more edge points
	/* This threshold eliminates responses at edges.  A value of 0.08 means
	   that the ratio of the largest to smallest eigenvalues (principle
	   curvatures) is below 10.  A value of 0.14 means ratio is less than 5.
	   A value of 0.0 does not eliminate any responses.
	   Threshold at first octave is different.
	*/
	float  EdgeThresh  /*0.06*/;
	float  EdgeThresh1 /*0.08*/;
	float  TensorThresh  /*0.06*/; // Mariano Rodríguez

	/* OriBins gives the number of bins in the histogram (36 gives 10
	   degree spacing of bins).
	*/
	int OriBins  /*36*/;


	/* Size of Gaussian used to select orientations as multiple of scale
		 of smaller Gaussian in DOG function used to find keypoint.
		 Best values: 1.0 for UseHistogramOri = FALSE; 1.5 for TRUE.
	*/
	float OriSigma  /*1.5*/;


	/// Look for local (3-neighborhood) maximum with valuer larger or equal than OriHistThresh * maxval
	///  Setting one returns a single peak
	/* All local peaks in the orientation histogram are used to generate
	   keypoints, as long as the local peak is within OriHistThresh of
	   the maximum peak.  A value of 1.0 only selects a single orientation
	   at each location.
	*/
	float OriHistThresh  /*0.8*/;


	/// Feature vector is normalized to has euclidean norm 1.
	/// This threshold avoid the excessive concentration of information on single peaks
	/* Index values are thresholded at this value so that regions with
	   high gradients do not need to match precisely in magnitude.
	   Best value should be determined experimentally.  Value of 1.0
	   has no effect.  Value of 0.2 is significantly better.
	*/
	float  MaxIndexVal  /*0.2*/;


	/* This constant specifies how large a region is covered by each index
	   vector bin.  It gives the spacing of index samples in terms of
	   pixels at this scale (which is then multiplied by the scale of a
	   keypoint).  It should be set experimentally to as small a value as
	   possible to keep features local (good values are in range 3 to 5).
	*/
	int  MagFactor   /*3*/;


	/* Width of Gaussian weighting window for index vector values.  It is
	   given relative to half-width of index, so value of 1.0 means that
	   weight has fallen to about half near corners of index patch.  A
	   value of 1.0 works slightly better than large values (which are
	   equivalent to not using weighting).  Value of 0.5 is considerably
	   worse.
	*/
	float   IndexSigma  /*1.0*/;

	/* If this is TRUE, then treat gradients with opposite signs as being
	   the same.  In theory, this could create more illumination invariance,
	   but generally harms performance in practice.
	*/
	int  IgnoreGradSign  /*0*/;



	float MatchRatio  /*0.6*/;

	/*
	   In order to constrain the research zone for matches.
	   Useful for example when looking only at epipolar lines
	*/

	float MatchXradius /*= 1000000.0f*/;
	float MatchYradius /*= 1000000.0f*/;

	int noncorrectlylocalized;

	/*
	 * If TRUE, then a ROOT-SIFT like version will be provided as in:
	 * Arandjelović, R., & Zisserman, A. (2012, June). Three things everyone should know to improve object retrieval. In Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on (pp. 2911-2918). IEEE.
	 *
	 * This is to normalize with respect to the L1 norm and then apply sqrt()
	*/
	bool MODE_ROOT; /*true*/


	bool half_sift_trick; /*=false*/
	bool L2norm; /* =true... false = L2 Norm   */

};

//////////////////////////////////////////////


#endif