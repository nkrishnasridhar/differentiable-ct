"""
Compare trainable parameters and embedding lookup cost for nearest vs trilinear.

Run from the repo root: python scripts/count_params.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from project_main import MultiScaleVoxelModel, _build_model

N_DEPTH = 8
EMBEDDING_DIM = 1
SIZE = 2.0


def lookups_per_point(lookup_mode, n_depth=N_DEPTH):
    gathers_per_level = 1 if lookup_mode == 'nearest' else 8
    return n_depth * gathers_per_level


def dense_multiply_adds_per_point(n_depth=N_DEPTH, embedding_dim=EMBEDDING_DIM):
    concat_dim = n_depth * embedding_dim
    return concat_dim * 64 + 64


def main():
    nearest = MultiScaleVoxelModel(SIZE, N_DEPTH, EMBEDDING_DIM, lookup_mode='nearest')
    trilinear = MultiScaleVoxelModel(SIZE, N_DEPTH, EMBEDDING_DIM, lookup_mode='trilinear')
    _build_model(nearest)
    _build_model(trilinear)

    nearest_params = nearest.count_params()
    trilinear_params = trilinear.count_params()

    print('Parameter counts (n_depth=8, embedding_dim=1)')
    print(f'  nearest:   {nearest_params}')
    print(f'  trilinear: {trilinear_params}')
    print(f'  equal: {nearest_params == trilinear_params}')

    print('Embedding table lookups per query point')
    print(f'  nearest:   {lookups_per_point("nearest")}')
    print(f'  trilinear: {lookups_per_point("trilinear")}')

    macs = dense_multiply_adds_per_point()
    print(f'Dense multiply-adds per query point (both models): {macs}')


if __name__ == '__main__':
    main()
