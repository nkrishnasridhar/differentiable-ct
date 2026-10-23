# Differentiable CT Reconstruction

Neural volumetric reconstruction of CT volumes from X-ray projections (COSC440 project).
A `MultiScaleVoxelModel` (nearest or trilinear voxel lookup) is trained by differentiating
through the projection geometry to match measured attenuation.

## Requirements

- Python 3, TensorFlow, NumPy, scikit-image
- GPU recommended (`TIGREDataset` uses `device="cuda"`)

## Usage

```bash
python project_main.py
```

Edit the dataset path at the bottom of `project_main.py` (`data/ct_data/*.pickle`),
or call `main(dataset_path, epochs, n_points, n_rays, model='nearest'|'trilinear', ...)`.

Metrics are logged to `results/csv/`, reconstructed slices written to `data/out/`,
and optional checkpoints/resume via `checkpoint_dir` / `resume`.

## Layout

- `geometry.py` — cone-beam geometry + TIGRE dataset loader
- `project_main.py` — model, training loop, evaluation
- `notebooks/` — Colab / exploratory notebook
- `scripts/` — parameter counting and result plotting
- `report/` — LaTeX report and figures
