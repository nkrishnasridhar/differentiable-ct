import tensorflow as tf
import numpy as np
from project_main import *

if __name__ == "__main__":
    rays = tf.convert_to_tensor(np.array([[1., 0., 0., -1., 0.1, 0.1]]),
                                dtype=tf.float64)  # not realistic, just for the test
    near = np.float64(0.9)
    far = np.float64(1.1)

    # small test
    points, distances = rays_to_points(rays, 10, near, far)
    print("rays_to_points output:")
    print(points)
    print(distances)

    attenuations = tf.convert_to_tensor(np.array([[0.5, 0.3, 0.1]]), dtype=tf.float32)
    distances = tf.convert_to_tensor(np.array([0.9, 1.0, 1.1]), dtype=tf.float32)
    magnitudes = tf.convert_to_tensor(np.array([2.0]), dtype=tf.float32)  # not realistic, just for the test
    result = ray_attenuation(attenuations, distances, magnitudes, near, far)
    print("ray_attenuation output:")
    print(result)

    encoder = MultiScaleVoxelModel(size=2.0, n_depth=8, embedding_dim=3)
    test_values = tf.convert_to_tensor(np.array([[0.5, 0.5, 0.5], [0.52, 0.500001, 0.5]]), dtype=tf.float32)
    print("get_flattened_position output:")
    for depth in range(0, 8):
        print(encoder.get_flattened_position(test_values, depth))

    encoder2 = MultiScaleVoxelModel(size=2.0, n_depth=3, embedding_dim=3)

    test_values2 = tf.convert_to_tensor(np.array([[0.5, 0.25, 0.125]]), dtype=tf.float32)
    for depth in range(0, 8):
        print(encoder.get_flattened_position(test_values2, depth))

""" Expected output:
rays_to_points output:
tf.Tensor(
[[[ 0.1         0.09        0.09      ]
  [ 0.07777778  0.09222222  0.09222222]
  [ 0.05555556  0.09444444  0.09444444]
  [ 0.03333333  0.09666667  0.09666667]
  [ 0.01111111  0.09888889  0.09888889]
  [-0.01111111  0.10111111  0.10111111]
  [-0.03333333  0.10333333  0.10333333]
  [-0.05555556  0.10555556  0.10555556]
  [-0.07777778  0.10777778  0.10777778]
  [-0.1         0.11        0.11      ]]], shape=(1, 10, 3), dtype=float64)
tf.Tensor(
[0.9        0.92222222 0.94444444 0.96666667 0.98888889 1.01111111
 1.03333333 1.05555556 1.07777778 1.1       ], shape=(10,), dtype=float64)
ray_attenuation output:
tf.Tensor([0.12000003], shape=(1,), dtype=float32)
get_flattened_position output:
tf.Tensor([7 7], shape=(2,), dtype=int32)
tf.Tensor([42 42], shape=(2,), dtype=int32)
tf.Tensor([292 292], shape=(2,), dtype=int32)
tf.Tensor([2184 2184], shape=(2,), dtype=int32)
tf.Tensor([16912 16912], shape=(2,), dtype=int32)
tf.Tensor([133152 133153], shape=(2,), dtype=int32)
tf.Tensor([1056832 1056834], shape=(2,), dtype=int32)
tf.Tensor([8421504 8421509], shape=(2,), dtype=int32)
tf.Tensor([1], shape=(1,), dtype=int32)
tf.Tensor([6], shape=(1,), dtype=int32)
tf.Tensor([84], shape=(1,), dtype=int32)
tf.Tensor([584], shape=(1,), dtype=int32)
tf.Tensor([4368], shape=(1,), dtype=int32)
tf.Tensor([33824], shape=(1,), dtype=int32)
tf.Tensor([266304], shape=(1,), dtype=int32)
tf.Tensor([2113664], shape=(1,), dtype=int32)
"""