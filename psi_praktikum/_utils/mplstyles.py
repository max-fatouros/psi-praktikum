"""
https://matplotlib.org/stable/tutorials/introductory/customizing.html#defining-your-own-style
"""
from psi_praktikum._utils import paths

PAPER = [
    'seaborn-v0_8',
    paths.UTIL_DIR / 'paper.mplstyle',
]

POSTER = [
    PAPER,
    paths.UTIL_DIR / 'poster.mplstyle',
]
