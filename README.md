# optical_calculate

English | [中文](README.zh-CN.md)

A collection of Python scripts for superoscillatory lens research, including random ring-radius data generation, diffraction integrals, optical performance evaluation, random simulation, and GA/NSGA-II optimization experiments.

## Background and goals

A superoscillatory lens is designed to produce a locally concentrated field below the conventional diffraction limit, enabling smaller focal spots or stronger local energy confinement. Here, a lens is represented by ring radii `rm`. The search therefore aims to find a sequence satisfying target field performance under physical and manufacturing constraints.

The research goal is a terahertz lens with a concentrated sagittal field and small focal spot, together with a long focal length and depth of focus in the meridional plane. Potential applications include high-resolution terahertz CT imaging.

## Features

- Generate training, validation, and test datasets of RM ring radii.
- Calculate focal length, FWHM, compression ratio, and sidelobe ratio using diffraction integrals.
- Run high-precision random simulations and export CSV files and plots.
- Search for candidate sequences using genetic algorithms and NSGA-II.
- Plot two-dimensional propagation intensity for representative RM sequences.

## Installation

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start

```powershell
# Generate a small dataset
python scripts/generate_rm_dataset.py --num-samples 100 --output-dir data/rm_dataset

# Calculate optical parameters
python scripts/compute_optical_params.py --input data/rm_dataset/train_dataset.csv --output outputs/optical_params_results.csv --processes 2

# Run a small random simulation
python scripts/simulation.py --num-samples 100 --output-dir outputs

# Plot an NSGA-II propagation example
python scripts/optics_nsga2.py --mode plot --output-dir outputs
```

## Scripts

- `scripts/generate_rm_dataset.py`: Generate a basic RM dataset.
- `scripts/generate_rm_dataset_v2.py`: Generate a dataset with control over the proportion of `r1=0` samples.
- `scripts/compute_optical_params.py`: Calculate optical parameters for each RM sequence in parallel.
- `scripts/diffraction_integral.py`: Analyze diffraction integrals for one or more sequences.
- `scripts/diffraction_integral_v2.py`: Diffraction analysis with peak and FWHM measurements.
- `scripts/simulation.py`: Run high-precision random simulations and save qualifying results.
- `scripts/optics_ga.py`: Genetic algorithm experiments.
- `scripts/optics_nsga2.py`: NSGA-II multi-objective optimization and propagation plots.
- `scripts/dna.py`: Differential evolution experiments.

## Data and results

Full datasets, model files, and generated results are not committed. Use the scripts to regenerate data; outputs are written to `outputs/` by default.

## License

MIT License.
