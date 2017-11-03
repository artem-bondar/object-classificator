#ifndef IO_PARSER_H_
#define IO_PARSER_H_

#include "classifier.h"

void LoadFileList(const string& data_file, TFileList* file_list);
void LoadImages(const TFileList& file_list, TDataSet* data_set);
void SavePredictions(const TFileList& file_list,
	const TLabels& labels,
	const string& prediction_file);

#endif
