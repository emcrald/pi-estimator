# CUDA Monte Carlo π Estimator

This project uses CUDA and PyCUDA to estimate the value of π using the Monte Carlo method. It launches GPU threads to rapidly generate random points and determine how many fall inside a unit circle.

## Features

- Utilizes PyCUDA for GPU acceleration.
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
git clone https://github.com/your-username/pi-calculator.git
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

Your Name – [@emcrald](https://github.com/emcrald)

````

---

### 📄 `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
.venv/
.env
*.pyo
*.pyd

# VS Code
.vscode/

# OS
.DS_Store
Thumbs.db
````

---

### 📄 `LICENSE`

```text
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...

[Standard MIT license text continues here, or copy from https://opensource.org/licenses/MIT]
```

---