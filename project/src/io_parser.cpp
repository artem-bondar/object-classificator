#include <fstream>
#include <iostream>

#include "io_parser.h"

using std::ifstream;
using std::ofstream;
using std::make_pair;
using std::endl;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
	ifstream stream(data_file.c_str());

	string filename;
	int label;

	int char_idx = data_file.size() - 1;
	for (; char_idx >= 0; --char_idx)
		if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
			break;
	string data_path = data_file.substr(0, char_idx + 1);

	while (!stream.eof() && !stream.fail()) {
		stream >> filename >> label;
		if (filename.size())
			file_list->push_back(make_pair(data_path + filename, label));
	}

	stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
	for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
		// Create image
		BMP* image = new BMP();
		// Read image from file
		image->ReadFromFile(file_list[img_idx].first.c_str());
		// Add image and it's label to dataset
		data_set->push_back(make_pair(image, file_list[img_idx].second));
	}
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
	const TLabels& labels,
	const string& prediction_file) {
	// Check that list of files and list of labels has equal size 
	assert(file_list.size() == labels.size());
	// Open 'prediction_file' for writing
	ofstream stream(prediction_file.c_str());

	// Write file names and labels to stream
	for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
		stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
	stream.close();
}
