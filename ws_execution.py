from __future__ import print_function
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import importlib
import time
import copy
import ctypes
from WeightSeparation import VNN
from WeightSeparation import WeightSeparation

tf.logging.set_verbosity(tf.logging.ERROR)

ws_op = tf.load_op_library('./tf_operation.so')
_weight_loader = ctypes.CDLL('./weight_loader.so')
_weight_loader.get_weight.argtypes = (ctypes.POINTER(ctypes.c_int64),
	ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
	ctypes.c_int, ctypes.c_int, ctypes.c_int64,
	ctypes.c_int64, ctypes.c_int, ctypes.c_int)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.060)
gpu_options = None

def init_separation(ws, vnn_list):
	with tf.Graph().as_default() as graph:
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

			time1 = time.time()
			shared_weight_address = sess.run(ws_op.init_weight(ws.weight_page))
			time2 = time.time()
			print('shared_weight GPU address: 0x%X' % shared_weight_address)
			print('init shared_weight %0.3f ms' % ((time2-time1)*1000.0))

			page_address_list = []
			exclusive_weight_page_list = []
			vnn_no = 0
			for vnn in vnn_list:
				page_list = copy.deepcopy(vnn.weight_page_list).tolist()
				vnn.exclusive_weight_page_list.sort()
				exclusive_page_list = copy.deepcopy(vnn.exclusive_weight_page_list)

				idx = 0
				for exclusive_page in exclusive_page_list:
					page_list.insert(exclusive_page, ws.num_of_weight_page + idx)
					idx += 1

				time1 = time.time()
				page_address = sess.run(ws_op.init_page_table(page_list))
				time2 = time.time()
				print('[VNN %d][%s] init page table %0.3f ms'
					% (vnn_no, vnn.name, (time2-time1)*1000.0))
				page_address_list.append(page_address)

				exclusive_weight_page = []
				for i in vnn.exclusive_weight_page_list:
					exclusive_weight_page.extend(ws.exclusive_weight_page[(vnn.name,i)])
				exclusive_weight_page_list.append(exclusive_weight_page)
				vnn_no += 1

			page_table_address_list = []
			for i in range(len(page_address_list)):
				page_table_address = tf.constant(page_address_list[i],
					name='page_table_address/' + str(i))
				page_table_address_list.append(page_table_address)

	return shared_weight_address, page_address_list, exclusive_weight_page_list

def load_weight_page(shared_weight_address, weight_address_list,
	weight_len_list, page_address, exclusive_weight_page,
	weight_per_page, num_of_weight_page):
	num_of_weight = len(weight_address_list)
	weight_address_list_array_type = ctypes.c_int64 * num_of_weight
	weight_len_list_array_type = ctypes.c_int * num_of_weight
	num_of_exclusive_weight_page = len(exclusive_weight_page)
	exclusive_weight_page_array_type = ctypes.c_float * num_of_exclusive_weight_page
	_weight_loader.get_weight(
		weight_address_list_array_type(*weight_address_list),
		weight_len_list_array_type(*weight_len_list),
		exclusive_weight_page_array_type(*exclusive_weight_page),
		ctypes.c_int(num_of_exclusive_weight_page),
		ctypes.c_int(num_of_weight),
		ctypes.c_int64(shared_weight_address),
		ctypes.c_int64(page_address),
		ctypes.c_int(weight_per_page),
		ctypes.c_int(num_of_weight_page))

def ws_execution(ws, vnn, shared_weight_address, page_address, exclusive_weight_page):
	print("[Executing]", vnn.name)

	with tf.Graph().as_default() as graph:
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

			saver = tf.train.import_meta_graph(vnn.meta_filepath)
			sess.run(tf.global_variables_initializer())

			train_weights = tf.trainable_variables()
			weight_address, weight_len = sess.run(ws_op.get_weight_address(train_weights))

			time1 = time.time()
			load_weight_page(shared_weight_address, weight_address,
				weight_len, page_address, exclusive_weight_page,
				ws.weight_per_page, ws.num_of_weight_page)
			time2 = time.time()
			weights_load_time = (time2-time1)*1000.0
			print('weights load time : %0.3f ms' % (weights_load_time))

			pintle = ws.import_pintle(vnn).pintle

			input_variable_names = pintle.v_input_variable_names()
			input_tensors = []
			for variable_name in input_variable_names:
				input_tensor_name = variable_name + ':0'
				input_tensors.append(graph.get_tensor_by_name(input_tensor_name))

			raw_input_variables = pintle.v_test_input_variables()
			input_variables = [ raw_input_variables[0][0] ]
			for i in range(1, len(raw_input_variables)):
				input_variables.append(raw_input_variables[i])
			ground_truth = raw_input_variables[0][1]

			output_variable_names = pintle.v_output_variable_names()
			output_tensors = []
			for variable_name in output_variable_names:
				output_tensor_name = variable_name + ':0'
				output_tensors.append(graph.get_tensor_by_name(output_tensor_name))

			time1 = time.time()
			result, accuracy = pintle.v_execute(graph, sess,
				input_tensors, input_variables, output_tensors,
				ground_truth=ground_truth)
			time2 = time.time()
			DNN_execution_time = (time2-time1)*1000.0
			print('DNN execution time: %0.3f ms' % (DNN_execution_time))

	return weights_load_time, DNN_execution_time, result, accuracy

def main():
	ws = WeightSeparation()

	vnn_list = []
	for name, vnn in sorted(ws.vnns.items()):
		vnn_list.append(vnn)

	total_weight_load_time = 0
	total_execution_time = 0
	num_execution = 50

	shared_weight_address, page_address_list, exclusive_weight_page_list \
		= init_separation(ws, vnn_list)

	for i in range(num_execution):
		vnn_no = np.random.randint(len(vnn_list))
		vnn = vnn_list[vnn_no]

		print("[%d/%d] total_weight_load_time: %0.3f ms, total_execution_time: %0.3f ms" % (i, num_execution, total_weight_load_time, total_execution_time))

		weight_load_time, execution_time, reuslt, accuracy = ws_execution(ws, vnn,
			shared_weight_address, page_address_list[vnn_no],
			exclusive_weight_page_list[vnn_no])

		total_weight_load_time += weight_load_time
		total_execution_time += execution_time

	print('total weights load time : %0.3f ms' % (total_weight_load_time))
	print('total DNN execution time: %0.3f ms' % (total_execution_time))

if __name__ == '__main__':
	main()
