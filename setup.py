import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
  requirements = f.read().splitlines()
setup(
  name='d4ft',
  version='0.0.1',
  packages=find_packages(),
  install_requires=requirements,
  extras_require={
    'dev': ['isort', 'mypy', 'yapf'],
  }
)
