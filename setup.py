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
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version='0.4.0',
    platforms=['mac', 'unix'],
    packages=find_packages(),
    install_requires=['h5py', 'matplotlib', 'scipy',
                      'scikit-image', 'cytoolz',
                      'tqdm', 'statsmodels', 'numpy',
                      'joblib', 'scikit-learn',
                      'click', 'ruamel.yaml',
                      'moseq2-pca @ git+https://github.com/dattalab/moseq2-pca.git@code_refactor',
                      'moseq2-model @ git+https://github.com/dattalab/moseq2-model.git@code_refactor',
                      'moseq2-viz @ git+https://github.com/dattalab/moseq2-viz.git@code_refactor'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-extract = moseq2_extract.cli:cli']}
)
