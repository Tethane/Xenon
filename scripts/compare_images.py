#!/usr/bin/env python3
"""Compare scene-linear PFM images; optionally save the same display transform."""

import argparse
import json
from pathlib import Path
import numpy as np
from benchmark import read_pfm, compare


def display(image, exposure):
	x = np.maximum(image * 2**exposure, 0)
	x = x / (1 + x)
	return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("candidate", type=Path)
	parser.add_argument("reference", type=Path)
	parser.add_argument("--output", type=Path)
	parser.add_argument("--figure", type=Path)
	parser.add_argument("--exposure", type=float, default=0)
	args = parser.parse_args()
	candidate, reference = read_pfm(args.candidate), read_pfm(args.reference)
	result = {
		"candidate": str(args.candidate.resolve()),
		"reference": str(args.reference.resolve()),
		"linear_metrics": compare(candidate, reference),
	}
	text = json.dumps(result, indent=2) + "\n"
	print(text, end="")
	if args.output:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(text)
	if args.figure:
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

		fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
		for ax, pixels, title in zip(
			axes[:2], [candidate, reference], [args.candidate.stem, args.reference.stem]
		):
			ax.imshow(display(pixels, args.exposure))
			ax.set_title(title)
			ax.axis("off")
		# Use an explicit shared scale for scientific error inspection.
		error = np.sqrt(np.mean((candidate - reference) ** 2, axis=2))
		im = axes[2].imshow(
			error, cmap="magma", vmin=0, vmax=max(float(np.quantile(error, 0.99)), 1e-8)
		)
		axes[2].set_title("Linear RGB RMSE per pixel (99th percentile scale)")
		axes[2].axis("off")
		fig.colorbar(im, ax=axes[2], fraction=0.045)
		fig.suptitle(
			f"Same exposure/Reinhard/sRGB transform; relative linear RMSE {result['linear_metrics']['relative_rmse']:.3%}"
		)
		args.figure.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(args.figure, dpi=140)
		plt.close(fig)


if __name__ == "__main__":
	main()
