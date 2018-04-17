#!/bin/bash

# get the latest miniconda for linux, make executable, put in path
curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/miniconda3_latest.sh"
chmod +x $HOME/miniconda3_latest.sh
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
