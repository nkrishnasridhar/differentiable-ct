"""
Regenerate report figures from committed CSV logs.

Run from the repo root: python scripts/plot_results.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_NEAREST = os.path.join('results', 'csv', 'nearest_chest_seed0.csv')
DEFAULT_TRILINEAR = os.path.join('results', 'csv', 'trilinear_chest_seed0.csv')
DEFAULT_OUT_DIR = os.path.join('report', 'figures')


def _run_label(df):
    row = df.iloc[0]
    return (
        f"chest, seed={int(row['seed'])}, "
        f"n_points={int(row['n_points'])}, n_rays={int(row['n_rays'])}"
    )


def plot_figure1_baseline(nearest_df, out_path):
    fig, (ax_loss, ax_ssim) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(nearest_df['epoch'], nearest_df['loss'], color='C0')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Normalised MSE loss')
    ax_loss.set_title('Baseline (nearest lookup)')
    ax_loss.set_xlim(nearest_df['epoch'].min(), nearest_df['epoch'].max())
    ax_loss.grid(True, alpha=0.3)

    ax_ssim.plot(nearest_df['epoch'], nearest_df['ssim'], color='C0')
    ax_ssim.set_xlabel('Epoch')
    ax_ssim.set_ylabel('Volumetric SSIM')
    ax_ssim.set_title('Baseline (nearest lookup)')
    ax_ssim.set_xlim(nearest_df['epoch'].min(), nearest_df['epoch'].max())
    ax_ssim.grid(True, alpha=0.3)

    fig.suptitle(_run_label(nearest_df), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_figure2_comparison(nearest_df, trilinear_df, out_path):
    fig, (ax_loss, ax_ssim) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(nearest_df['epoch'], nearest_df['loss'], label='nearest', color='C0')
    ax_loss.plot(trilinear_df['epoch'], trilinear_df['loss'], label='trilinear', color='C1')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Normalised MSE loss')
    ax_loss.legend()
    ax_loss.set_xlim(1, max(nearest_df['epoch'].max(), trilinear_df['epoch'].max()))
    ax_loss.grid(True, alpha=0.3)

    ax_ssim.plot(nearest_df['epoch'], nearest_df['ssim'], label='nearest', color='C0')
    ax_ssim.plot(trilinear_df['epoch'], trilinear_df['ssim'], label='trilinear', color='C1')
    ax_ssim.set_xlabel('Epoch')
    ax_ssim.set_ylabel('Volumetric SSIM')
    ax_ssim.legend()
    ax_ssim.set_xlim(1, max(nearest_df['epoch'].max(), trilinear_df['epoch'].max()))
    ax_ssim.grid(True, alpha=0.3)

    fig.suptitle(_run_label(nearest_df), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot report figures from training CSVs.')
    parser.add_argument('--nearest', default=DEFAULT_NEAREST)
    parser.add_argument('--trilinear', default=DEFAULT_TRILINEAR)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    nearest_df = pd.read_csv(args.nearest)
    trilinear_df = pd.read_csv(args.trilinear)

    fig1_path = os.path.join(args.out_dir, 'fig1_baseline.pdf')
    fig2_path = os.path.join(args.out_dir, 'fig2_comparison.pdf')

    plot_figure1_baseline(nearest_df, fig1_path)
    plot_figure2_comparison(nearest_df, trilinear_df, fig2_path)

    print(f'Wrote {fig1_path}')
    print(f'Wrote {fig2_path}')


if __name__ == '__main__':
    main()
