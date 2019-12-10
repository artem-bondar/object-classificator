#include <cmath>

#include "classifier_trainer.h"
#include "image_processing.h"
#include "io_parser.h"

HOG get_HOG_from_sector(const Matrix<double> &lengths, const Matrix<double> &directions) {
	HOG hog(PI_SPLIT, .0);

	for (uint i = 0; i < lengths.n_rows; i++) {
		for (uint j = 0; j < lengths.n_cols; j++) {
			hog[static_cast<uint>(((directions(i, j) + M_PI) / (2 * M_PI)) * PI_SPLIT) % PI_SPLIT] += static_cast<float>(lengths(i, j));
		}
	}

	return hog;
}

HOG extract_descriptor_from(BMP &image) {
	uint xSectorSize, ySectorSize,
		 xDelta, yDelta,
		 xSize, ySize;
	double hog_norm;
	Image img, grad_X, grad_Y;
	HOG descriptor, sector_hog;


	img = BMPtoImage(image);
	img = add_border(img, 1);
	img = mirror_border(img, 1);
	img = to_grayscale(img);

	grad_X = sobel_x(img);
	grad_Y = sobel_y(img);
	grad_X = cut_border(grad_X, 1);
	grad_Y = cut_border(grad_Y, 1);

	img = cut_border(img, 1);

	Matrix<double> lengths(img.n_rows, img.n_cols), directions(img.n_rows, img.n_cols);

	for (uint i = 0; i < img.n_rows; ++i) {
		for (uint j = 0; j < img.n_cols; ++j) {
			lengths(i, j) = std::sqrt(std::pow(std::get<0>(grad_X(i, j)), 2) + std::pow(std::get<0>(grad_Y(i, j)), 2));
			directions(i, j) = std::atan2(std::get<0>(grad_Y(i, j)), std::get<0>(grad_X(i, j)));
		}
	}

	xSectorSize = img.n_rows / X_SPLIT;
	ySectorSize = img.n_cols / Y_SPLIT;
	xDelta = img.n_rows % X_SPLIT;
	yDelta = img.n_cols % Y_SPLIT;

	for (uint i = 0; i < X_SPLIT; ++i) {
		xSize = (i == X_SPLIT - 1) ? xSectorSize + xDelta : xSectorSize;
		for (uint j = 0; j < Y_SPLIT; ++j) {
			ySize = (j == Y_SPLIT - 1) ? ySectorSize + yDelta : ySectorSize;
			sector_hog = get_HOG_from_sector(lengths.submatrix(i * xSectorSize, j * ySectorSize, xSize, ySize),
										  directions.submatrix(i * xSectorSize, j * ySectorSize, xSize, ySize));
			hog_norm = 0;
			for (uint k = 0; k < sector_hog.size(); ++k)
				hog_norm += std::pow(sector_hog[k], 2);
			hog_norm = std::sqrt(hog_norm);
			if (hog_norm > 0)
			{
				for (uint k = 0; k < sector_hog.size(); ++k)
					sector_hog[k] /= hog_norm;
			}
			descriptor.insert(descriptor.end(), sector_hog.begin(), sector_hog.end());
		}
	}

	return descriptor;
}

HOG get_LBP_from_sector(const Image &image) {
	HOG lbp(256, .0);

	for (uint i = 0; i < image.n_rows; i++) {
		for (uint j = 0; j < image.n_cols; j++) {
			lbp[std::get<0>(image(i, j))]++;
		}
	}

	return lbp;
}

HOG extract_local_binary_patterns_from(BMP &image) {
	uint xSectorSize, ySectorSize,
		xDelta, yDelta,
		xSize, ySize;
	double hog_norm;
	Image img;
	HOG lbp, sector_lbp;

	img = BMPtoImage(image);

	img = add_border(img, 1);
	img = mirror_border(img, 1);
	img = to_grayscale(img);
	img = img.unary_map(LocalBinaryPatternsFilter());
	img = cut_border(img, 1);

	xSectorSize = img.n_rows / X_LBP_SPLIT;
	ySectorSize = img.n_cols / Y_LBP_SPLIT;
	xDelta = img.n_rows % X_LBP_SPLIT;
	yDelta = img.n_cols % Y_LBP_SPLIT;

	for (uint i = 0; i < X_LBP_SPLIT; ++i) {
		xSize = (i == X_LBP_SPLIT - 1) ? xSectorSize + xDelta : xSectorSize;
		for (uint j = 0; j < Y_LBP_SPLIT; ++j) {
			ySize = (j == Y_LBP_SPLIT - 1) ? ySectorSize + yDelta : ySectorSize;
			sector_lbp = get_LBP_from_sector(img.submatrix(i * xSectorSize, j * ySectorSize, xSize, ySize));
			hog_norm = 0;
			for (uint k = 0; k < sector_lbp.size(); ++k)
				hog_norm += std::pow(sector_lbp[k], 2);
			hog_norm = std::sqrt(hog_norm);
			if (hog_norm > 0)
			{
				for (uint k = 0; k < sector_lbp.size(); ++k)
					sector_lbp[k] /= hog_norm;
			}
			lbp.insert(lbp.end(), sector_lbp.begin(), sector_lbp.end());
		}
	}

	return lbp;
}

HOG get_color_from_sector(const Image &image) {
	HOG color;

	uint size = image.n_rows * image.n_cols * 255;
	double R, G, B, SR, SG, SB;

	SR = SG = SB = 0;

	for (uint i = 0; i < image.n_rows; i++) {
		for (uint j = 0; j < image.n_cols; j++) {
			std::tie(R, G, B) = image(i, j);
			SR += std::get<0>(image(i, j));
			SG += std::get<1>(image(i, j));
			SB += std::get<2>(image(i, j));
		}
	}

	SR /= size;
	SG /= size;
	SB /= size;

	color.push_back(SR);
	color.push_back(SG);
	color.push_back(SB);

	return color;
}

HOG extract_color_features_from(BMP &image) {
	uint xSectorSize, ySectorSize,
		xDelta, yDelta,
		xSize, ySize;
	Image img;
	HOG color, sector_color;

	img = BMPtoImage(image);

	xSectorSize = img.n_rows / X_COLOR_SPLIT;
	ySectorSize = img.n_cols / Y_COLOR_SPLIT;
	xDelta = img.n_rows % X_COLOR_SPLIT;
	yDelta = img.n_cols % Y_COLOR_SPLIT;

	for (uint i = 0; i < X_COLOR_SPLIT; ++i) {
		xSize = (i == X_COLOR_SPLIT - 1) ? xSectorSize + xDelta : xSectorSize;
		for (uint j = 0; j < Y_COLOR_SPLIT; ++j) {
			ySize = (j == Y_COLOR_SPLIT - 1) ? ySectorSize + yDelta : ySectorSize;
			sector_color = get_color_from_sector(img.submatrix(i * xSectorSize, j * ySectorSize, xSize, ySize));
			color.insert(color.end(), sector_color.begin(), sector_color.end());
		}
	}

	return color;
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
	// Delete all images from dataset
	for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
		delete (*data_set)[image_idx].first;
	// Clear dataset
	data_set->clear();
}

// Extract features from dataset.
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
	HOG image_features, LBP, color_features;
	for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
		image_features = extract_descriptor_from(*(data_set[image_idx].first));

		LBP = extract_local_binary_patterns_from(*(data_set[image_idx].first));
		image_features.insert(image_features.end(), LBP.begin(), LBP.end());

		color_features = extract_local_binary_patterns_from(*(data_set[image_idx].first));
		image_features.insert(image_features.end(), color_features.begin(), color_features.end());

		features->push_back(make_pair(image_features, data_set[image_idx].second));
	}
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
	// List of image file names and its labels
	TFileList file_list;
	// Structure of images and its labels
	TDataSet data_set;
	// Structure of features of images and its labels
	TFeatures features;
	// Model which would be trained
	TModel model;
	// Parameters of classifier
	TClassifierParams params;

	// Load list of image file names and its labels
	LoadFileList(data_file, &file_list);
	// Load images
	LoadImages(file_list, &data_set);
	// Extract features from images
	ExtractFeatures(data_set, &features);

	// You can change parameters of classifier here
	params.C = 0.01;
	TClassifier classifier(params);
	// Train classifier
	classifier.Train(features, &model);
	// Save model to file
	model.Save(model_file);
	// Clear dataset structure
	ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
	const string& model_file,
	const string& prediction_file) {
	// List of image file names and its labels
	TFileList file_list;
	// Structure of images and its labels
	TDataSet data_set;
	// Structure of features of images and its labels
	TFeatures features;
	// List of image labels
	TLabels labels;

	// Load list of image file names and its labels
	LoadFileList(data_file, &file_list);
	// Load images
	LoadImages(file_list, &data_set);
	// Extract features from images
	ExtractFeatures(data_set, &features);

	// Classifier 
	TClassifier classifier = TClassifier(TClassifierParams());
	// Trained model
	TModel model;
	// Load model from file
	model.Load(model_file);
	// Predict images by its features using 'model' and store predictions
	// to 'labels'
	classifier.Predict(features, model, &labels);

	// Save predictions
	SavePredictions(file_list, labels, prediction_file);
	// Clear dataset structure
	ClearDataset(&data_set);
}
