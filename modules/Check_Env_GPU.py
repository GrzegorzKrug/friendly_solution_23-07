# import tensorflow as tf
# import tensorflow.keras
import keras
import tensorflow as tf
import tensorflow.keras as k2

# from tensorflow.

# from keras import tensorflow as tf

# assert tf.test.is_gpu_available(), "NO GPU"
# print(f"GPU: {tf.test.is_gpu_available()}")
# print(f"GPU: {tf.test.is_gpu_available(cuda_only=False)}")
# print(f"GPU: {tf.test.is_gpu_available(min_cuda_compute_capability=False)}")
# print(f"GPU: {tf.test.is_gpu_available(min_cuda_compute_capability=True)}")

print("CPU LIST:", tf.config.list_physical_devices("CPU"))
print("GPU LIST:", tf.config.list_physical_devices("GPU"))
# print("GPU AVAILABLE:", tf.test.is_gpu_available()) # Deprecated
print("Deprecated AVAILABLE:", tf.test.is_gpu_available())  # Deprecated
print("Deprecated AVAILABLE (noCuda):", tf.test.is_gpu_available(cuda_only=False))  # Deprecated
print("Deprecated AVAILABLE (Cuda):", tf.test.is_gpu_available(cuda_only=True))  # Deprecated
print("BUILD WITH CUDA:", tf.test.is_built_with_cuda())  # Installed non gpu package

from tensorflow.python.client import device_lib


print("=== " * 6)
print("LOCAL DEVICES:")
print(device_lib.list_local_devices())
