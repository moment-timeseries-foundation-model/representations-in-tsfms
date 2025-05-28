from setuptools import setup, find_packages

setup(
    name="steertool",
    version="0.1.0",
    packages=["steertool"],
    install_requires=[
        "numpy",
        "torch",
        "pandas",
        "pyyaml",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "joblib",
        "tqdm",
        "momentfm",
        "nnsight",
    ],
    entry_points={
        "console_scripts": [
            "steertool=steertool.cli:main",
        ],
    },
) 