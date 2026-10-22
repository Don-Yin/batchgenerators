"""Measure returned image and segmentation sizes from real transform calls."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from batchgenerators.augmentations.spatial_transformations import augment_resize, augment_zoom
from batchgenerators.transforms import spatial_transforms
from batchgenerators.transforms.spatial_transforms import ResizeTransform, ZoomTransform


def measure():
    records = []
    for operation in ("resize", "zoom"):
        for container in (list, tuple):
            sample = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
            segmentation = (sample > 5).astype(np.uint8)
            if operation == "resize":
                transform = ResizeTransform((2, 2), order=0, order_seg=0)
                expected = augment_resize(sample, segmentation, (2, 2), 0, 0)
            else:
                transform = ZoomTransform(0.5, order=0, order_seg=0)
                expected = augment_zoom(sample, segmentation, 0.5, 0, 0)
            result = transform(data=container([sample]), seg=container([segmentation]))
            for key, original, reference in zip(("data", "seg"), (sample, segmentation), expected):
                actual = result[key][0]
                records.append({
                    "operation": operation,
                    "container": container.__name__,
                    "field": key,
                    "input_shape": list(original.shape),
                    "expected_shape": list(reference.shape),
                    "actual_shape": list(actual.shape),
                    "input_values": original.tolist(),
                    "expected_values": reference.tolist(),
                    "actual_values": actual.tolist(),
                    "matches_expected": bool(np.array_equal(actual, reference)),
                })
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "transform_sha256": hashlib.sha256(Path(spatial_transforms.__file__).read_bytes()).hexdigest(),
        "input_kind": "synthetic arrays; all outputs measured by calling the transforms",
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    arguments.output.write_text(json.dumps(measure(), indent=2) + "\n")
