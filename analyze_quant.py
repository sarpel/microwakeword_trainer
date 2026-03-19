import math
import sys

import tensorflow as tf


def logit(p):
    return math.log(p / (1 - p))


model_path = sys.argv[1]
model_name = model_path.split("/")[-1].replace(".tflite", "")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()

logit_tensor_index = None
prob_tensor_index = None

for i, detail in enumerate(tensor_details):
    if detail["name"] == "StatefulPartitionedCall:01":
        logit_tensor_index = i
    elif detail["name"] == "StatefulPartitionedCall:0":
        prob_tensor_index = i

if logit_tensor_index is None or prob_tensor_index is None:
    print(f"Could not find tensors in {model_name}")
    sys.exit(1)

details_logit = tensor_details[logit_tensor_index]
details_prob = tensor_details[prob_tensor_index]

scale_logit = details_logit["quantization_parameters"]["scales"][0]
zp_logit = details_logit["quantization_parameters"]["zero_points"][0]

scale_prob = details_prob["quantization_parameters"]["scales"][0]
zp_prob = details_prob["quantization_parameters"]["zero_points"][0]

print(f"Model: {model_name}")
print(f"Logit tensor: scale={scale_logit}, zero_point={zp_logit}")
print(f"Prob tensor: scale={scale_prob}, zero_point={zp_prob}")

# Compute INT8 for probs
probs = [0.5, 0.9, 0.95, 0.97, 0.99]
for p in probs:
    x = logit(p)
    q = round(x / scale_logit + zp_logit)
    print(f"p={p}: logit={x:.3f}, INT8={q}")

# Number of levels in [0.90, 0.97]
q90 = round(logit(0.9) / scale_logit + zp_logit)
q97 = round(logit(0.97) / scale_logit + zp_logit)
levels = abs(q97 - q90) + 1
print(f"Distinct INT8 levels in [0.90, 0.97]: {levels}")
print()
