import tensorflow as tf

pb_file = "E:\\spleeter_freeze_v1.pb"
input_arrays = ["mix_stft"]
output_arrays = ["mul_2", "mul_3"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(pb_file, input_arrays, output_arrays, {"mix_stft": [1, 2049, 2]})
tflite_model = converter.convert()
open("E:\\spleeter_v1.tflite", "wb").write(tflite_model)
