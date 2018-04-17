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
# get the latest miniconda for linux, make executable, put in path
curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/miniconda3_latest.sh"
# make it executable
chmod +x $HOME/miniconda3_latest.sh
# install miniconda with no user intervention, then append to PATH
$HOME/miniconda3_latest.sh -b -p $HOME/miniconda3
cat >> ~/.bashrc << END
# add for miniconda install
PATH=\$HOME/miniconda3/bin:\$PATH
END

# make sure we have the conda command available
source $HOME/.bashrc
conda create -n "moseq2" python=3.6 -y
source activate moseq2

# now install our python code
mkdir $HOME/python_repos
git clone https://github.com/dattalab/moseq2.git $HOME/python_repos/moseq2
pip install -e $HOME/python_repos/moseq2
```

If you get any errors with respect to LibSM, e.g. in CentOS in an HPC environment, you may need to run this command *before* installing via pip,

```sh
sudo yum install libSM
```

## Usage

Most usage patterns are available via passing `--help` on the command line.  To see the available command,

```sh
moseq2 --help
```

Then to see the options available for each command, e.g.,

```sh
moseq2 extract --help
```

### Extracting data (interactive)

To extract data, simply point `moseq2 extract` to any `depth.dat` file,

```sh
moseq2 extract ~/my_awesome_data/depth.data
```

This automatically select an ROI and extract data to the `proc` folder where `depth.dat` is located.

### Extracting data (batch)

For batch extractions, see [moseq2-batch](https://github.com/dattalab/moseq2-batch). 

### Flip classification

Under construction

## Support

## Contributing

If you would like to contribute, fork the repository and issue a pull request.  
