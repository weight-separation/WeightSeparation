from __future__ import print_function
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import importlib
import time
import ctypes
from WeightSeparation import VNN
from WeightSeparation import WeightSeparation

tf.logging.set_verbosity(tf.logging.ERROR)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.060)
gpu_options = None

def baseline_execution(tlwv, vnn):
	print("[Executing]", vnn.name)

	with tf.Graph().as_default() as graph:
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

			saver = tf.train.import_meta_graph(vnn.meta_filepath)
			sess.run(tf.global_variables_initializer())

			time1 = time.time()
			saver.restore(sess, vnn.model_filepath)
			time2 = time.time()
			weights_load_time = (time2-time1)*1000.0
			print('weights load time : %0.3f ms' % (weights_load_time))

			pintle = tlwv.import_pintle(vnn).pintle

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
	tlwv = WeightSeparation()

	vnn_list = []
	for name, vnn in sorted(tlwv.vnns.items()):
		vnn_list.append(vnn)

	total_weight_load_time = 0
	total_execution_time = 0
	num_execution = 50

	for i in range(num_execution):
		vnn_no = np.random.randint(len(vnn_list))
		vnn = vnn_list[vnn_no]

		print("[%d/%d] total_weight_load_time: %0.3f ms, total_execution_time: %0.3f ms" % (i, num_execution, total_weight_load_time, total_execution_time))

		weight_load_time, execution_time, result, accuracy = baseline_execution(tlwv, vnn)
	
		total_weight_load_time += weight_load_time
		total_execution_time += execution_time

	print('total weights load time : %0.3f ms' % (total_weight_load_time))
	print('total DNN execution time: %0.3f ms' % (total_execution_time))

if __name__ == '__main__':
	main()
