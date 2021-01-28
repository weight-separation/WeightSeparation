from __future__ import division, print_function, unicode_literals
import numpy as np
import struct
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

train_data_file = os.path.join(dir_path, 'obstacle_train_data.npy')
if os.path.exists(train_data_file):
	obstacle_train_data = np.load(train_data_file)
train_label_file = os.path.join(dir_path, 'obstacle_train_label.npy')
if os.path.exists(train_label_file):
	obstacle_train_label = np.load(train_label_file)
test_data_file = os.path.join(dir_path, 'obstacle_test_data.npy')
if os.path.exists(test_data_file):
	obstacle_test_data = np.load(test_data_file)
test_label_file = os.path.join(dir_path, 'obstacle_test_label.npy')
if os.path.exists(test_label_file):
	obstacle_test_label = np.load(test_label_file)
validation_data_file = os.path.join(dir_path, 'obstacle_validation_data.npy')
if os.path.exists(validation_data_file):
	obstacle_validation_data = np.load(validation_data_file)
validation_label_file = os.path.join(dir_path, 'obstacle_validation_label.npy')
if os.path.exists(validation_label_file):
	obstacle_validation_label = np.load(validation_label_file)

def train_set():
	return obstacle_train_data, obstacle_train_label

def test_set():
	return obstacle_test_data, obstacle_test_label

def validation_set():
	return obstacle_validation_data, obstacle_validation_label

def create_data_files():
	obstacle_train_data, obstacle_train_label = get_cifar_train_batch()
	obstacle_test_data, obstacle_test_label = get_cifar_test_batch()

	np.save('obstacle_train_data', obstacle_train_data)
	print (obstacle_train_data.shape)
	np.save('obstacle_train_label', obstacle_train_label)
	print (obstacle_train_label.shape)
	np.save('obstacle_test_data', obstacle_test_data)
	print (obstacle_test_data.shape)
	np.save('obstacle_test_label', obstacle_test_label)
	print (obstacle_test_label.shape)

def write_data():
	f = open('obs_t', 'w')
	print(obstacle_test_data.shape)
	data_len = obstacle_test_data.shape[0]
	test_data = obstacle_test_data
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
	f = open('obs_l', 'w')
	data_len = obstacle_test_label.shape[0]
	single_data_size = 1
	print('data_len:', data_len)
	print('single_data_size:', single_data_size)

	f.write(struct.pack('i', data_len))
	f.write(struct.pack('i', single_data_size))

	for label in obstacle_test_label:
		f.write(struct.pack('b', np.argmax(label)))
	f.close()

def main():
	print('main')
	#write_data()
	#write_label()

if __name__ == '__main__':
	main()
