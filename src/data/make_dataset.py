# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

DATA_DIR = '../../data/raw'
PROCESSED_DATA_DIR = '../../data/processed'

def load_csv_file(filename):
	return pd.read_csv(os.path.join(DATA_DIR, filename))

def encode_categorical_to_numerical(train, test):
	train_ = train.copy()
	test_ = test.copy()

	encoders_train = {}
	encoders_test = {}

	for column in train.columns[:-1]:
		if train[column].dtype == np.object:
			lbl = LabelEncoder()
			encoders_train[column] = lbl.fit_transform(list(train[column]) + list(test[column]))
			train_[column] = lbl.transform(train[column])
		
			lbl = LabelEncoder()
			encoders_test[column] = lbl.fit_transform( list(train[column]) + list(test[column]))
			test_[column] = lbl.transform(test[column])

	return train_, test_, encoders_train, encoders_test	

def fill_missing_values(train, test):
	train['Product_Category_2'] = train.Product_Category_2.fillna(8.0) # 8.0 is the most common value
	train['Product_Category_3'] = train.Product_Category_3.fillna(16.0) # 16.0 is the most common value

	test['Product_Category_2'] = test.Product_Category_2.fillna(8.0) # 8.0 is the most common value
	test['Product_Category_3'] = test.Product_Category_3.fillna(16.0) # 16.0 is the most common value

	return train, test

def save_processed_file(df, filename):
	df.to_csv(os.path.join(PROCESSED_DATA_DIR, filename), index=False)

def main(train_input_filepath, test_input_filepath, output_train_filepath, output_test_filepath):
	train = load_csv_file(train_input_filepath)
	test = load_csv_file(test_input_filepath)
	
	train_, test_, encoders_train, encoders_test = encode_categorical_to_numerical(train, test)

	train_, test_ = fill_missing_values(train_, test_)
	
	save_processed_file(train_, output_train_filepath)
	save_processed_file(test_, output_test_filepath)

if __name__ == '__main__':
	train_input_filepath = 'train.csv'
	test_input_filepath = 'test-comb.csv'
	output_train_filepath = 'train_processed.csv'
	output_test_filepath = 'test_processed.csv'

	main(train_input_filepath, test_input_filepath, output_train_filepath, output_test_filepath)
