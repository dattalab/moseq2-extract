from setuptools import setup, find_packages
import subprocess
import sys
import os


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


try:
    import cv2
except ImportError:
    install('opencv-python')

try:
    import cython
except ImportError:
    install('cython')

os.system('export CC="$(which gcc-7)"')
os.system('export CXX="$(which g++-7)"')

# note that scikit-learn must == 0.19 since flip classifiers were trained using this version
setup(
    name='moseq2-extract',
    author='Datta Lab',
    description='To boldly go where no mouse has gone before',
    version='0.4.4',
    platforms=['mac', 'unix'],
    packages=find_packages(),
    install_requires=['h5py==2.10.0', 'tqdm==4.40.0', 'scipy==1.4.1', 'numpy==1.18.3', 'click==7.0',
                      'joblib==0.14.0', 'cytoolz==0.10.1', 'matplotlib==3.1.2', 'statsmodels==0.10.2',
                      'scikit-image==0.16.2', 'scikit-learn==0.22', 'opencv-python==4.1.2.30',
                      'ruamel.yaml==0.16.5'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-extract = moseq2_extract.cli:cli']}
)