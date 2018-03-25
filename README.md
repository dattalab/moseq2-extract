# MoSeq2 [![Build Status](https://travis-ci.com/dattalab/moseq2.svg?token=gvoikVySDHEmvHT7Dbed&branch=master)](https://travis-ci.com/dattalab/moseq2) [![codecov](https://codecov.io/gh/dattalab/moseq2/branch/master/graph/badge.svg?token=ICPjpMMwYZ)](https://codecov.io/gh/dattalab/moseq2)

Welcome to moseq2, the latest version of a software package for mouse tracking in depth videos first developed by Alex Wiltschko in the Datta Lab at Harvard Medical School.

## Table of Contents  

- [Installation](#installation)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)

## Installation

Moseq has been tested with Python 3.4 and later.  All requirements can be installed using pip, though it's recommended you create a new conda environment to ensure the dependencies do not conflict with any software you may have currently installed.

```sh
conda create -n moseq2 python=3.6
source activate moseq2
git clone https://github.com/dattalab/moseq2.git
cd moseq2/
pip install -e .
moseq2-model --help
```

If you get any errors with respect to LibSM, e.g. in CentOS in an HPC environment, you may need to run this command *before* installing via pip,

```sh
sudo yum install libSM
```

## Usage

```python
moseq2 --help
```

## Support

## Contributing
