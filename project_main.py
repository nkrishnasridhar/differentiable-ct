import csv
import os
import subprocess
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from geometry import TIGREDataset
import skimage.io

def rays_to_points(rays, n_points, near, far):
    """
    Computes the sample points for the given rays. For the base implementation, sample uniform distances such
    that the distance between each point is (far-near)/(n_points-1). For later implementations you may want to change
    it to produce perturbed points with varying distances.
    To get a point along a ray, you want to take each ray's directional vector, multiply by the distance for that
     point along the ray, and then add it to the ray origin.

    :param rays: tensor of float32 [N, 6] for N rays defined by 6 values: A 3D point for the origin of the ray and a 3D vector of the direction of the ray
    :param n_points: int, the number of points to sample along each ray
    :param near: float, the closest distance to scale the ray by
    :param far: float, the farthest distance to scale the ray by
    :return: a two element tuple: (points, distances) where
      points is [N, n_points, 3] for N rays with n_points each of 3D points
      distances is [n_points] for the distance along the ray for each point
    """
    dtype = rays.dtype
    near = tf.cast(near, dtype)
    far = tf.cast(far, dtype)
    distances = tf.linspace(near, far, n_points)
    origins = rays[:, :3]
    directions = rays[:, 3:6]
    t = tf.reshape(distances, [1, -1, 1])
    points = tf.expand_dims(origins, axis=1) + t * tf.expand_dims(directions, axis=1)
    return points, distances


def ray_attenuation(attenuations, distances, magnitudes, near, far):
    """
    Computes the sum of attenuation for each sampled set of attenuations given the distances
     of the points along the source to detector axis and magnitudes of the vectors.

    A fully correct trapezoidal integration is to create starting / ending points for
    near and far along the ray, then compute distances between all subsequent points along the ray,
     then interpolate the attenuations between points, and finally create a weighted sum and multiply
     it by the magnitude.

    :param attenuations: tensor of floats [n_rays, n_points] where n_rays is the number of rays per image used and n_points is the number of points per ray used
    :param distances: tensor of floats [n_points] which is the distance along each ray in the source to detector axis
    :param magnitudes: tensor of floats [n_rays] which is the magnitude of each directional ray
    :param near: float, the closest distance to region of interest
    :param far: float, the farthest distance to region of interest
    :return: tensor of floats [n_rays] which is the attenuation value for each ray
    """
    dt = tf.reshape(distances[1:] - distances[:-1], [1, -1])
    segment = (attenuations[:, :-1] + attenuations[:, 1:]) * 0.5 * dt
    integral = tf.reduce_sum(segment, axis=1)
    return integral * magnitudes

class MultiScaleVoxelModel(tf.keras.Model):
    """
    MultiScaleVoxelModel uses keras EmbeddingLayer instances for the core 3D Model.
    It has n_depth number of layers with each having storage for a power of 2
    hierarchical decomposition of 3D space, i.e.

    layer 0 = an embedding layer for a 2x2x2 grid which is 8 total
    layer 1 = an embedding layer for a 4x4x4 grid which is 64 total
    layer 2 = an embedding layer for a 8x8x8 grid which is 512 total
    ... for n_depth layers total (0 based so with n_depth=8 the last layer would be layer 7 with 256x256x256 = 16777216)

    The input dimension of an embedding layer is the same range as the get_flattened_position function returns,
    so calling get_flattened_position(value, 8) for the above network would produce a flattened position in the range
    [0, 16777216)

    Each element stores embedding_dim values (1 by default).
    Output from the embedding layers is combined using a simple 2 layer Dense network with silu activation.
    """

    def __init__(self, size, n_depth, embedding_dim = 1):
        super(MultiScaleVoxelModel, self).__init__()

        self.size = size
        self.n_depth = n_depth
        self.embedding_dim = embedding_dim

        # 1. Multi-scale embedding tables (2x2x2, 4x4x4, 8x8x8...)
        self.embedding_layers = []
        for embedding_i in range(n_depth):
            grid_resolution = 2 ** (embedding_i + 1)
            table_capacity = grid_resolution ** 3

            emb = keras.layers.Embedding(
                table_capacity,
                embedding_dim
                #embeddings_constraint=tf.keras.constraints.NonNeg()
            )
            self.embedding_layers.append(emb)
        self.hidden_dense = tf.keras.layers.Dense(64, activation='silu')

        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def get_flattened_position(self, scaled_values, depth):
        """
        Take in a 3-dimensional position and map it onto multiple 3-D grids, where each grid is the original grid divided
         by a factor of 2

        For example, in a 4x4x4 grid, there are 3 multi-resolutions -- 1x1x1, 2x2x2, 4x4x4
        We will ignore the 1x1x1 dimension because all points would have position 0.
        If the query point came in as 0.5, 0.25, 0.125 for this grid, the positional encoder should produce two values:
        1 (position of the point in a 2x2x2 grid which has 8 positions total)
        6 (position of the point in a 4x4x4 grid which has 64 positions total)

        Input: tensor of float32, shape [N, 3] of N points in 3-D space normalized such that all points are [0,1)
        Output: tensor of int32, shape [N] of N flattened positions
        """
        resolution = 2 ** (depth + 1)
        scaled = scaled_values * tf.cast(resolution, scaled_values.dtype)
        ix = tf.cast(tf.floor(scaled[:, 0]), tf.int32)
        iy = tf.cast(tf.floor(scaled[:, 1]), tf.int32)
        iz = tf.cast(tf.floor(scaled[:, 2]), tf.int32)
        r = tf.constant(resolution, dtype=tf.int32)
        return ix + r * iy + r * r * iz

    def call(self, inputs):
        embeddings_list = []
        # the input positions are between -self.size / 2 and self.size / 2
        scaled_values = tf.clip_by_value(0.5 + inputs / tf.cast(self.size, tf.float32), 0.0, 1.0 - 1e-7)
        # scaled_values are now [0,1)
        
        # Collect features from all resolutions separately
        for depth in range(self.n_depth):
            idx = self.get_flattened_position(scaled_values, depth)
            features = self.embedding_layers[depth](idx)
            embeddings_list.append(features)

        # Combine features using the two Dense layers
        concatenated_features = tf.concat(embeddings_list, axis=-1)
        x = self.hidden_dense(concatenated_features)
        return self.output_layer(x)

@tf.function
def train(model, dataset, optimizer, n_points):
    """
    Simple training loop that iterates through each projection image, samples rays from that image,
    sends the points of those rays through the network, computes the predicted attenuation per ray,
    computes the loss between the predicted value and the true value, and then updates the network.
    """
    num_projections = dataset.rays.shape[-1]
    total_loss = 0.0
    permutation = tf.random.shuffle(tf.range(num_projections))

    # 🌟 2. Loop using the shuffled indices
    for idx in permutation:
        projection, rays = dataset[idx]
        points, distances = rays_to_points(rays, n_points, dataset.near, dataset.far)
        magnitudes = tf.norm(rays[..., 3:6], axis=-1)
        n_rays = points.shape[0]
        points = tf.reshape(points, (-1, 3))

        with tf.GradientTape() as tape:
            attenuation = model(points)
            attenuation = tf.reshape(attenuation, (n_rays, -1))
            predicted_attenuation = ray_attenuation(attenuation, distances, magnitudes, dataset.near, dataset.far)
            loss = tf.keras.losses.MSE(projection, predicted_attenuation)
            normalized_loss = loss / tf.reduce_mean(tf.square(projection))
            total_loss += normalized_loss
        gradients = tape.gradient(normalized_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss / num_projections

METRICS_CSV_FIELDS = [
    'epoch', 'loss', 'ssim', 'psnr', 'pred_max_before_norm', 'frac_negative',
    'epoch_seconds', 'ssim_scope', 'git_commit',
    'model', 'dataset', 'seed', 'n_points', 'n_rays', 'n_depth', 'embedding_dim', 'learning_rate',
]


def _dataset_name(dataset_path):
    base = os.path.splitext(os.path.basename(dataset_path))[0]
    if base.endswith('_50'):
        return base[:-3]
    return base


def _git_commit_short():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def append_metrics_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def get_sample_slices(model, dataset):
    """
    Queries the network at many 3-dimensional points to produce a voxelized grid of attenuation values.
    Note that the returned values are scaled by the max value.
    """
    slices = dataset.voxels.shape[2]
    slice_list = []

    for i in range(slices):
        voxels = tf.convert_to_tensor(dataset.voxels[:, :, i], dtype=tf.float32)
        shape = voxels.shape[0:2]
        voxels = tf.reshape(voxels, (-1, 3))

        # 3. Predict the clean slice
        image_pred = model(voxels)
        image_pred = tf.reshape(image_pred, shape)
        slice_list.append(image_pred)

    uint8_volume, _, _ = _volume_from_slice_predictions(slice_list)
    return uint8_volume


def _volume_from_slice_predictions(slice_list):
    imarr = np.array(slice_list, dtype=np.float32)
    pred_max = float(np.max(imarr))
    frac_negative = float(np.mean(imarr < 0.0))
    clipped = np.maximum(imarr, 0.0)
    scale = np.max(clipped)
    if scale > 0:
        uint8_volume = ((clipped / scale) * 255).astype(np.uint8)
    else:
        uint8_volume = np.zeros(clipped.shape, dtype=np.uint8)
    return uint8_volume, pred_max, frac_negative


def evaluate_volume_metrics(model, dataset, ground_truth_volume):
    slices = dataset.voxels.shape[2]
    slice_list = []
    for i in range(slices):
        voxels = tf.convert_to_tensor(dataset.voxels[:, :, i], dtype=tf.float32)
        shape = voxels.shape[0:2]
        voxels = tf.reshape(voxels, (-1, 3))
        image_pred = model(voxels)
        slice_list.append(tf.reshape(image_pred, shape).numpy())

    predicted_volume, pred_max, frac_negative = _volume_from_slice_predictions(slice_list)
    pred_expanded = tf.expand_dims(predicted_volume, axis=-1)
    gt_expanded = tf.expand_dims(ground_truth_volume, axis=-1)
    psnr_per_slice = tf.image.psnr(pred_expanded, gt_expanded, max_val=255)
    ssim_per_slice = tf.image.ssim(pred_expanded, gt_expanded, max_val=255)
    volumetric_ssim = float(tf.reduce_mean(ssim_per_slice).numpy())
    volumetric_psnr = float(tf.reduce_mean(psnr_per_slice).numpy())
    return predicted_volume, volumetric_ssim, volumetric_psnr, pred_max, frac_negative


def main(dataset_path, epochs, n_points, n_rays, save_gt=False, save_projections=False, seed=0, log_csv=None):
    """
    Loads the data, saves a ground truth image, and then creates the model.
    Runs for a given number of epochs and number of sample points/sample rays for each projection image training loop.
    Saves a TIFF image of the sample slice output every 10 epochs.
    """
    tf.keras.utils.set_random_seed(seed)

    dataset = TIGREDataset(dataset_path, device="cuda", n_rays=n_rays)

    # need to transpose to get top down view
    ground_truth_volume = (dataset.ground_truth.transpose((2,0,1))*255).astype(np.uint8)
    if save_gt:
        skimage.io.imsave(f'data/out/gt.tiff', ground_truth_volume)
    if save_projections:
        os.makedirs('data/out/projections', exist_ok=True)
        for i, projection in enumerate(dataset.projections):
            p = projection.numpy()
            arr = ((p / np.max(p))*255).astype(np.uint8)
            skimage.io.imsave(f'data/out/projections/{i}.tiff', arr)

    n_depth = 8
    embedding_dim = 1
    learning_rate = 0.1
    model_name = 'nearest'
    dataset_name = _dataset_name(dataset_path)
    run_id = f'{model_name}_{dataset_name}_seed{seed}'
    csv_path = log_csv if log_csv is not None else os.path.join('results', 'csv', f'{run_id}.csv')
    git_commit = _git_commit_short()

    size = dataset.far - dataset.near
    model = MultiScaleVoxelModel(size, n_depth, embedding_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    print(f'Starting training... logging to {csv_path}')
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        epoch_loss = train(model, dataset, optimizer, n_points)
        epoch_seconds = time.perf_counter() - t0

        predicted_volume, volumetric_ssim, volumetric_psnr, pred_max, frac_negative = (
            evaluate_volume_metrics(model, dataset, ground_truth_volume)
        )
        loss_val = float(epoch_loss.numpy())

        print(
            f'Epoch {epoch} loss: {loss_val} SSIM: {volumetric_ssim} PSNR: {volumetric_psnr}'
        )

        append_metrics_row(csv_path, {
            'epoch': epoch,
            'loss': loss_val,
            'ssim': volumetric_ssim,
            'psnr': volumetric_psnr,
            'pred_max_before_norm': pred_max,
            'frac_negative': frac_negative,
            'epoch_seconds': epoch_seconds,
            'ssim_scope': 'full',
            'git_commit': git_commit,
            'model': model_name,
            'dataset': dataset_name,
            'seed': seed,
            'n_points': n_points,
            'n_rays': n_rays,
            'n_depth': n_depth,
            'embedding_dim': embedding_dim,
            'learning_rate': learning_rate,
        })

        if epoch % 10 == 0:
            os.makedirs('data/out', exist_ok=True)
            skimage.io.imsave(f'data/out/{epoch}.tiff', predicted_volume)

if __name__ == '__main__':
    dataset_path = 'data/ct_data/chest_50.pickle'
    # dataset_path = 'data/ct_data/abdomen_50.pickle'
    # dataset_path = 'data/ct_data/foot_50.pickle'
    # dataset_path = 'data/ct_data/jaw_50.pickle' # note this dataset is not as nice/clean

    # 50 epochs is not enough to produce a high quality reconstruction but you should see
    # a clear shape after 10 epochs

    # NOTE: you can pass flags for save_gt and save_projections if you want to look at the
    #  raw projections your model gets as input and the ground truth volume it compares to
    main(dataset_path, epochs=50, n_points=192, n_rays=2048)

