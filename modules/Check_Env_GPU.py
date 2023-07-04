import tensorflow as tf
import tensorflow.keras


# assert tf.test.is_gpu_available(), "NO GPU"
# print(f"GPU: {tf.test.is_gpu_available()}")
# print(f"GPU: {tf.test.is_gpu_available(cuda_only=False)}")
# print(f"GPU: {tf.test.is_gpu_available(min_cuda_compute_capability=False)}")
# print(f"GPU: {tf.test.is_gpu_available(min_cuda_compute_capability=True)}")

print("CPU LIST:", tf.config.list_physical_devices("CPU"))
print("GPU LIST:", tf.config.list_physical_devices("GPU"))
# print("GPU AVAILABLE:", tf.test.is_gpu_available()) # Deprecated
print("BUILD WITH CUDA:", tf.test.is_built_with_cuda())  # Installed non gpu package

from tensorflow.python.client import device_lib


print("LOCAL DEVICES:")
print(device_lib.list_local_devices())
