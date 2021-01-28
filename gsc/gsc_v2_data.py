from __future__ import print_function
import numpy as np
import os
import struct

dir_path = os.path.dirname(os.path.realpath(__file__))

train_data_file = os.path.join(dir_path, 'gsc_v2_train_data.npy')
if os.path.exists(train_data_file):
	gsc_v2_train_data = np.load(train_data_file)
train_label_file = os.path.join(dir_path, 'gsc_v2_train_label.npy')
if os.path.exists(train_label_file):
	gsc_v2_train_label = np.load(train_label_file)
test_data_file = os.path.join(dir_path, 'gsc_v2_test_data.npy')
if os.path.exists(test_data_file):
	gsc_v2_test_data = np.load(test_data_file)
test_label_file = os.path.join(dir_path, 'gsc_v2_test_label.npy')
if os.path.exists(test_label_file):
	gsc_v2_test_label = np.load(test_label_file)
validation_data_file = os.path.join(dir_path, 'gsc_v2_validation_data.npy')
if os.path.exists(validation_data_file):
	gsc_v2_validation_data = np.load(validation_data_file)
validation_label_file = os.path.join(dir_path, 'gsc_v2_validation_label.npy')
if os.path.exists(validation_label_file):
	gsc_v2_validation_label = np.load(validation_label_file)

def train_set():
	return gsc_v2_train_data, gsc_v2_train_label

def test_set():
	return gsc_v2_test_data, gsc_v2_test_label

def validation_set():
	return gsc_v2_validation_data, gsc_v2_validation_label

def write_data():
	f = open('gsc_t', 'w')
	print(gsc_v2_test_data.shape)
	data_len = gsc_v2_test_data.shape[0]
	test_data = gsc_v2_test_data
	print(test_data.shape)
	single_data_size = np.prod(test_data.shape[1:])
	print('data_len:', data_len)
	print('single_data_size:', single_data_size)

	f.write(struct.pack('i', data_len))
	f.write(struct.pack('i', single_data_size))

	for data in test_data:
		flattened = data.flatten()
		for i in range(len(flattened)):
			f.write(struct.pack('f', flattened[i]))
	f.close()

def write_label():
	f = open('gsc_l', 'w')
	data_len = gsc_v2_test_label.shape[0]
	single_data_size = 1
	print('data_len:', data_len)
	print('single_data_size:', single_data_size)

	f.write(struct.pack('i', data_len))
	f.write(struct.pack('i', single_data_size))

	for label in gsc_v2_test_label:
		f.write(struct.pack('b', np.argmax(label)))
	f.close()

def main():
	print('main')
	#write_data()
	#write_label()

if __name__ == '__main__':
        main()
