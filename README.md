# CUDA Monte Carlo π Estimator

This project uses CUDA and PyCUDA to estimate the value of π using the Monte Carlo method. It launches GPU threads to rapidly generate random points and determine how many fall inside a unit circle.

## Features

- Utilises PyCUDA for GPU acceleration.
- Estimates π with hundreds of millions of points in under 10 seconds.
- Demonstrates use of CURAND on the GPU.

## Requirements

- Python 3.7+
- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with 12.9)
- PyCUDA

## Setup

1. Clone the repo:

```bash
git clone https://github.com/emcrald/pi-calculator.git
cd pi-calculator
````

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
. .venv/Scripts/activate  # On Windows
pip install pycuda numpy
```

3. Run the script:

```bash
python script.py
```

## Output Example

```bash
Estimated π: 3.141543
Points: 500000000
Time taken: 9.90 seconds
```

## How it Works

* Each GPU thread generates a random `(x, y)` point inside the unit square.
* It checks if the point falls within the unit circle.
* The ratio of points inside the circle to the total gives an approximation of π.

## License

MIT – see [LICENSE](./LICENSE)

## Author

Emerald – [@emcrald](https://github.com/emcrald)