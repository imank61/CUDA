#include "sift_cuda.cuh"


__global__ void MakeKeypointKernel(siftPar* param, keypointHolder* holder, keypoint* key, float* grad, float* ori)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= param->totalCount)
		return;

	auto& h = holder[i];
	auto& k = key[i];
	int goIdx = h.imOffset;

	k.x = h.octSize * h.octCol;	/*x coordinate */
	k.y = h.octSize * h.octRow;	/*y coordinate */
	k.scale = h.octSize * h.octScale;	/* scale */
	k.angle = h.angle;		/* orientation */

	MakeKeypointSampleKernel(k, &grad[goIdx], &ori[goIdx], h.imWidth, h.imHeight, h.octScale, h.octRow, h.octCol, *param);
}

__device__ void MakeKeypointSampleKernel(
	keypoint& key, const float* grad, const float* ori,
	int nwidth, int nheight, float scale, float row, float
	col, siftPar& par)
{
	KeySampleVecKernel(key, grad, ori, nwidth, nheight, scale, row, col, par);

	NormalizeVecKernel(key.vec);

	bool changed = false;
	for (int i = 0; i < VecLength; i++)
	{
		if (key.vec[i] > par.MaxIndexVal)
		{
			key.vec[i] = par.MaxIndexVal;
			changed = true;
		}
	}

	if (changed)
		NormalizeVecKernel(key.vec);

	if (ACCURACY >= 1)
	{
		int intval;
		for (int i = 0; i < VecLength; i++)
		{
			intval = (int)(512.0 * key.vec[i]);
			key.vec[i] = (int)MIN(255, intval);
		}
	}
}

__device__ void KeySampleVecKernel(
	keypoint& key, const float* grad, const float* ori,
	int nwidth, int nheight, float scale, float row, float col, siftPar& par)
{
	memset(key.vec, 0, VecLength * sizeof(float));
	KeySampleKernel(key, grad, ori, nwidth, nheight, scale, row, col, par);
}

__device__ void KeySampleKernel(
	keypoint& key,
	const float* grad, const float* ori, int nwidth, int nheight,
	float scale, float row, float col, siftPar& par)
{
	float rpos, cpos, rx, cx;

	int	irow = (int)(row + 0.5),
		icol = (int)(col + 0.5);
	float	sine = (float)sin(key.angle),
		cosine = (float)cos(key.angle);

	float	spacing = scale * par.MagFactor;

	float	radius = 1.414 * spacing * (IndexSize + 1) / 2.0;
	int	iradius = (int)(radius + 0.5);

	float ts_xx = 0.0f, ts_yy = 0.0f, ts_xy = 0.0f;

	key.radius = (float)iradius; // Mariano Rodríguez

	for (int i = -iradius; i <= iradius; i++)
	{
		for (int j = -iradius; j <= iradius; j++)
		{
			rpos = ((cosine * i - sine * j) - (row - irow)) / spacing;
			cpos = ((sine * i + cosine * j) - (col - icol)) / spacing;

			rx = rpos + IndexSize / 2.0 - 0.5;
			cx = cpos + IndexSize / 2.0 - 0.5;

			if (rx > -1.0 && rx < (float)IndexSize &&
				cx > -1.0 && cx < (float)IndexSize)
			{
				int r = (int)(irow + i), c = (int)(icol + j);

				if (r >= 0 && r < nheight && c >= 0 && c < nwidth)
				{
					AddSampleKernel(key, grad, ori, nwidth, nheight, r, c, rpos, cpos, rx, cx, par);

					float mag = grad[r * nwidth + c];
					float theta = ori[r * nwidth + c];
					float dx = mag * cos(theta);
					float dy = mag * sin(theta);

					ts_xx += (dx * dx);
					ts_yy += (dy * dy);
					ts_xy += dx * dy;
				}
			}
		}
	}

	if (par.TensorThresh > 0.0)
	{
		float	det = ts_xx * ts_yy - ts_xy * ts_xy,	/// Det H = \prod l_i
			trace = ts_xx + ts_yy;		/// tr H = \sum l_i

		if ((det < par.TensorThresh * trace * trace))
			key.radius = -1.0f;
	}
}

__device__ void AddSampleKernel(
	keypoint& key,
	const float* grad, const float* orim, int nwidth, int nheight,
	int r, int c, float rpos, float cpos, float rx, float cx, siftPar& par)
{
	float	sigma = par.IndexSigma * 0.5 * IndexSize;
	float	weight = exp(-(rpos * rpos + cpos * cpos) / (2.0 * sigma * sigma));
	float	mag = weight * grad[r * nwidth + c];
	float	ori = orim[r * nwidth + c] - key.angle;

	if (par.IgnoreGradSign)
	{
		int nPI = int(abs(ori) / PI);

		if (ori > 0)
		{
			ori -= nPI * PI;
		}
		else if (ori < 0.0)
		{
			ori += nPI * PI;

			if (ori < 0.0)
				ori += PI;
		}
	}
	else
	{
		const double _2PI = 2.0 * PI;

		int n2PI = int(abs(ori) / _2PI);

		if (ori > 0)
		{
			ori -= n2PI * _2PI;
		}
		else if (ori < 0.0)
		{
			ori += n2PI * _2PI;

			if (ori < 0.0)
				ori += _2PI;
		}
	}

	PlaceInIndexKernel(key.vec, mag, ori, rx, cx, par);
}

__device__ void PlaceInIndexKernel(float* index,
	float mag, float ori, float rx, float cx, siftPar& par)
{
	int	orr, rindex, cindex, oindex;
	float	rweight, cweight, oweight;
	float* ivec;

	float	oval = OriSize * ori / (par.IgnoreGradSign ? PI : 2.0 * PI); /* grad quantize */

	int	ri = (int)((rx >= 0.0) ? rx : rx - 1.0),	/* Round down to next integer. */ // Guoshen Yu, explicitely cast to int to avoid warning
		ci = (int)((cx >= 0.0) ? cx : cx - 1.0), // Guoshen Yu, explicitely cast to int to avoid warning
		oi = (int)((oval >= 0.0) ? oval : oval - 1.0); // Guoshen Yu, explicitely cast to int to avoid warning

	float	rfrac = rx - ri,			/* Fractional part of location. */
		cfrac = cx - ci,
		ofrac = oval - oi;

	assert(
		ri >= -1 && ri < (int)IndexSize &&
		oi >= 0 && oi <= (int)OriSize &&
		rfrac >= 0.0 && rfrac <= 1.0);

	for (int r = 0; r < 2; r++)
	{
		rindex = ri + r;
		if (rindex >= 0 && rindex < (int)IndexSize)
		{
			rweight = mag * ((r == 0) ? 1.0 - rfrac : rfrac);

			for (int c = 0; c < 2; c++)
			{
				cindex = ci + c;
				if (cindex >= 0 && cindex < (int)IndexSize)
				{
					cweight = rweight * ((c == 0) ? 1.0 - cfrac : cfrac);
					ivec = &index[rindex * IndexSize * OriSize + cindex * OriSize];

					for (orr = 0; orr < 2; orr++)
					{
						oindex = oi + orr;

						if (oindex >= (int)OriSize)  /* Orientation wraps around at PI. */
							oindex = 0;

						oweight = cweight * ((orr == 0) ? 1.0 - ofrac : ofrac);
						ivec[oindex] += oweight;
					}
				}
			}
		}
	}
}

__device__ void NormalizeVecKernel(float* vec)
{
	float val, fac;

	float sqlen = 0.0;
	for (int i = 0; i < VecLength; i++)
	{
		val = vec[i];
		sqlen += val * val;
	}

	fac = 1.0 / sqrt(sqlen);

	for (int i = 0; i < VecLength; i++)
		vec[i] *= fac;
}


