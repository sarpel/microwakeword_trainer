import sys
import tensorflow as tf

model_path = sys.argv[1]
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()
for detail in tensor_details:
    print(f"Index: {detail['index']}, Name: {detail['name']}, Shape: {detail['shape']}, Dtype: {detail['dtype']}")
