from setuptools import setup, find_packages
# note that scikit-learn must == 0.19 since flip classifiers were trained using this version
setup(
    name='moseq2-extract',
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version='0.1.3',
    platforms=['mac', 'unix'],
    packages=find_packages(),
    install_requires=['h5py', 'matplotlib', 'scipy>=0.19',
                      'scikit-image>=0.13', 'scikit-learn==0.19',
                      'tqdm', 'statsmodels>=0.8.0', 'numpy==1.14.5',
                      'joblib==0.13.1', 'opencv-python>=4',
                      'click', 'ruamel.yaml>=0.15.0'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-extract = moseq2_extract.cli:cli']}
)
