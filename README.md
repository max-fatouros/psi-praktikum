# PSI Praktikum


## Installation

Using python3.10+, run

``` bash
pip install -e .
```


## Usage

Add functions to the `psi_praktikum/__main__.py`  file, then run those function in one of two ways.

### 1. As a script
Run

``` bash
python -m psi_praktikum
```

This will execute the `main()` function in the `psi_praktikum/__main__.py` file.


### 2. In a Jupyter notebook
See the example in [`notebooks/example.ipynb`](notebooks/example.ipynb)


## Project structure
Most of this is boilerplate code and configs from a python template I use.
The important files are listed below.

```
.
├── data/
│   └── PSI_lab_2025/  # a final copy of all the data we collected
├── notebooks/  # a place for Juypter notebooks
└── psi_praktikum
    ├── __main__.py
    └── _utils/
```
