#ifndef IMAGE_PROCESSING_H_
#define IMAGE_PROCESSING_H_

#include <tuple>

#include "EasyBMP.h"
#include "matrix.h"

Image BMPtoImage(BMP &image);
Image custom(Image src_image, Matrix<double> kernel);
Image sobel_x(Image src_image);
Image sobel_y(Image src_image);
Image add_border(Image src_image, const uint border_radius);
Image cut_border(Image src_image, const uint border_radius);
Image mirror_border(Image src_image, const uint border_radius);
Image to_grayscale(Image src_image);
template<typename T, typename R> void shoveInBounds(T& a, T& b, T& c, const R Min, const R Max) {
	T min = static_cast<T>(Min);
	T max = static_cast<T>(Max);
	a = (a > max) ? max : (a < min) ? min : a;
	b = (b > max) ? max : (b < min) ? min : b;
	c = (c > max) ? max : (c < min) ? min : c;
}

class ConvolutionFilter
{
public:
	ConvolutionFilter(const Matrix<double> &kernel);
	std::tuple<uint, uint, uint> operator()(const Image &filterZone) const;
	const uint radius;
private:
	Matrix<double> filterKernel;
};

class LocalBinaryPatternsFilter
{
public:
	LocalBinaryPatternsFilter();
	std::tuple<uint, uint, uint> operator()(const Image &filterZone) const;
	const uint radius;
};

#endif
