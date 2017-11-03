#include "image_processing.h"

using std::tuple;
using std::make_tuple;
using std::tie;

Image BMPtoImage(BMP &image)
{
	Image res(image.TellHeight(), image.TellWidth());

	for (uint i = 0; i < res.n_rows; ++i) {
		for (uint j = 0; j < res.n_cols; ++j) {
			RGBApixel *p = image(j, i);
			res(i, j) = make_tuple(p->Red, p->Green, p->Blue);
		}
	}
	return res;
}

Image custom(Image src_image, Matrix<double> kernel) {
	return src_image.unary_map(ConvolutionFilter(kernel));
}

Image sobel_x(Image src_image) {
	Matrix<double> kernel = { { -1, 0, 1 },
							  { -2, 0, 2 },
							  { -1, 0, 1 } };
	return custom(src_image, kernel);
}

Image sobel_y(Image src_image) {
	Matrix<double> kernel = { { 1,  2,  1 },
							  { 0,  0,  0 },
							  { -1, -2, -1 } };
	return custom(src_image, kernel);
}

ConvolutionFilter::ConvolutionFilter(const Matrix<double> &kernel) : radius(kernel.n_rows), filterKernel(kernel) {}

std::tuple<uint, uint, uint> ConvolutionFilter::operator()(const Image &filterZone) const
{
	double R, G, B, SR, SG, SB;

	SR = SG = SB = 0;
	for (uint i = 0; i < radius; ++i) {
		for (uint j = 0; j < radius; ++j) {
			std::tie(R, G, B) = filterZone(i, j);
			SR += R * filterKernel(i, j);
			SG += G * filterKernel(i, j);
			SB += B * filterKernel(i, j);
		}
	}

	shoveInBounds(SR, SG, SB, 0, 255);

	return std::make_tuple(SR, SG, SB);
}

LocalBinaryPatternsFilter::LocalBinaryPatternsFilter() : radius(3) {}

std::tuple<uint, uint, uint> LocalBinaryPatternsFilter::operator()(const Image &filterZone) const
{
	uint X = 0;
	std::vector<bool> patterns;

	for (uint i = 0; i < radius; ++i) {
		for (uint j = 0; j < radius; ++j) {
			patterns.push_back((filterZone(i,j) >= filterZone(radius / 2, radius / 2)));
		}
	}

	patterns.erase(patterns.begin() + radius);

	for (uint i = 0; i < patterns.size(); ++i) {
		X += patterns[i] * std::pow(2, i);
	}

	return std::make_tuple(X, X, X);
}

Image add_border(Image src_image, const uint border_radius) {
	Image new_image(src_image.n_rows + 2 * border_radius, src_image.n_cols + 2 * border_radius);
	for (uint i = 0; i < src_image.n_rows; ++i) {
		for (uint j = 0; j < src_image.n_cols; ++j) {
			new_image(i + border_radius, j + border_radius) = src_image(i, j);
		}
	}
	return new_image;
}

Image cut_border(Image src_image, const uint border_radius) {
	if (2 * border_radius + 1 > src_image.n_rows || 2 * border_radius + 1 > src_image.n_rows)
		throw std::string("radius is too big for this picture");
	Image new_image(src_image.n_rows - 2 * border_radius, src_image.n_cols - 2 * border_radius);
	for (uint i = 0; i < new_image.n_rows; ++i) {
		for (uint j = 0; j < new_image.n_cols; ++j) {
			new_image(i, j) = src_image(i + border_radius, j + border_radius);
		}
	}
	return new_image;
}

Image mirror_border(Image src_image, const uint border_radius) {
	// Mirrowing edge squares of image
	for (uint i = 0; i < border_radius; ++i) {
		for (uint j = 0; j < border_radius; ++j) {
			// Left upper
			src_image(i, j) = src_image(2 * border_radius - 1 - j, 2 * border_radius - 1 - i);
			// Right upper
			src_image(i, src_image.n_cols - 1 - j) = src_image(src_image.n_rows - 2 * border_radius + j, src_image.n_cols - 2 * border_radius + i);
			// Left down
			src_image(src_image.n_rows - 1 - i, j) = src_image(src_image.n_rows - 2 * border_radius + j, 2 * border_radius - 1 - i);
			// Right down
			src_image(src_image.n_rows - 1 - i, src_image.n_cols - 1 - j) = src_image(src_image.n_rows - 2 * border_radius + j, src_image.n_cols - 2 * border_radius + i);
		}
	}
	// Mirrowing side borders of image
	for (uint i = 0; i < src_image.n_rows - 2 * border_radius; ++i) {
		for (uint j = 0; j < border_radius; ++j) {
			// Left
			src_image(i + border_radius, j) = src_image(i + border_radius, 2 * border_radius - 1 - j);
			// Right
			src_image(i + border_radius, src_image.n_cols - 1 - j) = src_image(i + border_radius, src_image.n_cols - 2 * border_radius + j);
		}
	}
	for (uint i = 0; i < src_image.n_cols - 2 * border_radius; ++i) {
		for (uint j = 0; j < border_radius; ++j) {
			// Up
			src_image(j, i + border_radius) = src_image(2 * border_radius - 1 - j, i + border_radius);
			// Down
			src_image(src_image.n_rows - 1 - j, i + border_radius) = src_image(src_image.n_rows - 2 * border_radius + j, i + border_radius);
		}
	}
	return src_image;
}

Image to_grayscale(Image src_image)
{
	double R, G, B, Y;
	Image dst_image(src_image.n_rows, src_image.n_cols);

	for (uint i = 0; i < dst_image.n_rows; ++i) {
		for (uint j = 0; j < dst_image.n_cols; ++j) {
			tie(R, G, B) = src_image(i, j);
			Y = 0.299*R + 0.587*G + 0.114*B;
			Y = (Y > 255) ? 255 : (Y < 0) ? 0 : Y;
			dst_image(i, j) = make_tuple(Y, Y, Y);
		}
	}

	return dst_image;
}
