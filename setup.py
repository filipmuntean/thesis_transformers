from setuptools import setup, find_packages

setup(
    name = "bsc_proj§",
    version = "0.1",
    description = "Classification & Generator model of IMDB Movie reviews built with transformers",
    url = "https://github.com/filipmuntean/thesis_transformers/blob/main/setup.py",
    author = 'Filip Muntean',
    author_email = 'filipmorris@duck.com',
    license = 'MIT',
    package_dir = {"":"util"},
    packages = find_packages(where = "util"),
    install_requires = [
    "pytorch >= 1.10",
    'torch',
    'tqdm',
    'numpy',
    'torchtext',
    'tensorboard'],
    zip_safe = False
)
