# Sequence resize evidence

`resize-evidence.png` compares measured output side lengths for `ResizeTransform` and `ZoomTransform`, with list and tuple inputs, for both image data and segmentation.

- Input: one synthetic 4 × 4 sample and a binary segmentation.
- Requested output: 2 × 2, using `ResizeTransform((2, 2))` or `ZoomTransform(0.5)`.
- Both interpolation orders are zero; `concatenate_list` uses its default, `False`.
- Before: upstream commit `12a397217f95f62b706dd1be86b7b0f50cf24d1d`.
- After: fix commit `8e52e98ef93f8bc8cb86006d1d72ffc030479bc4`.
- Environment: Python 3.12.0, NumPy 2.4.2, SciPy 1.17.1, scikit-image 0.25.2.

`before.json` and `after.json` retain the input arrays, measured outputs, expected lower-level augmentation outputs, and source hashes. `measure.py` produces either JSON file when run from the corresponding source checkout with that checkout on `PYTHONPATH`:

```bash
PYTHONPATH=. python measure.py before.json
```

All plotted values come from transform calls on the synthetic inputs.
