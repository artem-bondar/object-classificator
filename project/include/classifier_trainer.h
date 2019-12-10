#ifndef CLASSIFIER_TRAINER_H_
#define CLASSIFIER_TRAINER_H_

#include "classifier.h"
#include "matrix.h"

// amount of parts to split source image
#define X_SPLIT 5
#define Y_SPLIT 5

// splits for binary patterns
#define X_LBP_SPLIT 7
#define Y_LBP_SPLIT 7

// splits for color features
#define X_COLOR_SPLIT 8
#define Y_COLOR_SPLIT 8

// amount of parts to split angle [-Pi, Pi]
#define PI_SPLIT 16

typedef std::vector<float> HOG;

HOG get_HOG_from_sector(const Matrix<double> &lengths, const Matrix<double> &directions);
HOG extract_descriptor_from(BMP &image);
HOG get_LBP_from_sector(const Image &image);
HOG extract_local_binary_patterns_from(BMP &image);
HOG get_color_from_sector(const Image &image);
HOG extract_color_features_from(BMP &image);
void ClearDataset(TDataSet* data_set);
void ExtractFeatures(const TDataSet& data_set, TFeatures* features);
void TrainClassifier(const string& data_file, const string& model_file);
void PredictData(const string& data_file,
	const string& model_file,
	const string& prediction_file);

#endif
