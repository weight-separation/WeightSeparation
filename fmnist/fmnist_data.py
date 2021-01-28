from __future__ import print_function
import numpy as np
import os
import struct

dir_path = os.path.dirname(os.path.realpath(__file__))

train_data_file = os.path.join(dir_path, 'fmnist_train_data.npy')
if os.path.exists(train_data_file): 
	fmnist_train_data = np.load(train_data_file)
train_label_file = os.path.join(dir_path, 'fmnist_train_label.npy')
if os.path.exists(train_label_file): 
	fmnist_train_label = np.load(train_label_file)
test_data_file = os.path.join(dir_path, 'fmnist_test_data.npy')
if os.path.exists(test_data_file): 
	fmnist_test_data = np.load(test_data_file)
test_label_file = os.path.join(dir_path, 'fmnist_test_label.npy')
if os.path.exists(test_label_file): 
	fmnist_test_label = np.load(test_label_file)

def train_set():
	return fmnist_train_data, fmnist_train_label

def test_set():
	return fmnist_test_data, fmnist_test_label

def validation_set():
	return None, None

def write_data():
	f = open('fmnist_t', 'w')
	print(fmnist_test_data.shape)
	data_len = fmnist_test_data.shape[0]
	test_data = fmnist_test_data
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
	f = open('fmnist_l', 'w')
	data_len = fmnist_test_label.shape[0]
	single_data_size = 1
	print('data_len:', data_len)
	print('single_data_size:', single_data_size)

	f.write(struct.pack('i', data_len))
	f.write(struct.pack('i', single_data_size))

	for label in fmnist_test_label:
		f.write(struct.pack('b', np.argmax(label)))
	f.close()

def main():
	print('main')
	#write_data()
	#write_label()

if __name__ == '__main__':
        main()
