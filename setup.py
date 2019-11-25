from setuptools import setup, find_packages
import subprocess
import sys


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


try:
    import cv2
except ImportError:
    install('opencv-python')

# note that scikit-learn must == 0.19 since flip classifiers were trained using this version
setup(
    name='moseq2-extract',
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version='0.2.3',
    platforms=['mac', 'unix'],
    packages=find_packages(),
    install_requires=['h5py', 'matplotlib', 'scipy',
                      'scikit-image', 'cytoolz',
                      'tqdm', 'statsmodels', 'numpy',
                      'joblib', 'scikit-learn',
                      'click', 'ruamel.yaml'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-extract = moseq2_extract.cli:cli']}
)
