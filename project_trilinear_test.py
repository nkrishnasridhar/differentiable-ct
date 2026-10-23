import tensorflow as tf

from project_main import (
    MultiScaleVoxelModel,
    _build_model,
    trilinear_corner_indices_and_weights,
    trilinear_embedding_lookup,
)

ATOL = 1e-6


def check_trilinear_matches_nearest_at_integer_coords():
    encoder = MultiScaleVoxelModel(size=2.0, n_depth=3, embedding_dim=3)
    _build_model(encoder)

    test_points = tf.constant(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ],
        dtype=tf.float32,
    )

    for depth in range(encoder.n_depth):
        embedding = encoder.embedding_layers[depth]
        nearest_idx = encoder.get_flattened_position(test_points, depth)
        nearest_features = embedding(nearest_idx)
        trilinear_features = trilinear_embedding_lookup(embedding, test_points, depth)
        max_diff = tf.reduce_max(tf.abs(nearest_features - trilinear_features))
        if float(max_diff) >= ATOL:
            raise AssertionError(
                f'depth {depth}: trilinear != nearest at integer coords, max_diff={float(max_diff)}'
            )


def check_half_cell_weights():
    half_cell = tf.constant([[0.375, 0.375, 0.375]], dtype=tf.float32)
    _, weights = trilinear_corner_indices_and_weights(half_cell, depth=1)
    weights = weights.numpy()[0]
    if not (abs(weights.sum() - 1.0) < ATOL and (abs(weights - 0.125) < ATOL).all()):
        raise AssertionError(f'expected eight weights of 0.125, got {weights}')


if __name__ == '__main__':
    check_trilinear_matches_nearest_at_integer_coords()
    print('integer coordinate check: passed')
    check_half_cell_weights()
    print('half-cell weights check: passed')
