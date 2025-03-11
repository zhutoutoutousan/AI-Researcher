from setuptools import setup, find_packages

setup(
    name="graph_learning",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torch-geometric',
        'numpy',
        'matplotlib',
        'seaborn',
        'pandas'
    ]
)