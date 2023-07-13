#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from pathlib import Path


def make_error_bins(labels, predictions, num_bins=100):
    bin_errors = np.zeros(num_bins, dtype=float)
    labels_discretized = np.digitize(labels, np.linspace(0, 1, num_bins + 1))
    for i in range(1, num_bins + 1):
        bin_errors[i - 1] = np.mean(
            np.abs(
                labels[labels_discretized == i] - predictions[labels_discretized == i]
            )
        )
    # replace nan values with the mean error value
    bin_errors[np.isnan(bin_errors)] = np.nanmean(bin_errors)
    bin_errors = (bin_errors - bin_errors.min()) / (bin_errors.max() - bin_errors.min())
    return bin_errors


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--num_bins", type=int, default=100)
    args = p.parse_args()

    print("Loading labels and predictions...")
    labels = np.load(args.labels)
    predictions = np.load(args.predictions)
    print(f"Calculating errors for {args.num_bins} bins...")
    error_bins = make_error_bins(labels, predictions, num_bins=args.num_bins)

    print(f"Saving results to {args.output_dir}/bin_errors.npy")
    np.save(args.output_dir / "bin_errors.npy", error_bins)
