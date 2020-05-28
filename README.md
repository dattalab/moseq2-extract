# MoSeq2-Extract: Depth Video Rodent-Tracking Toolkit
 
[![Build Status](https://travis-ci.com/dattalab/moseq2-extract.svg?token=gvoikVySDHEmvHT7Dbed&branch=master)](https://travis-ci.com/dattalab/moseq2-extract)
  
[![codecov](https://codecov.io/gh/dattalab/moseq2-extract/branch/master/graph/badge.svg?token=ICPjpMMwYZ)](https://codecov.io/gh/dattalab/moseq2-extract)

Welcome to moseq2, the latest version of a software package for mouse tracking in depth videos first developed by Alex Wiltschko in the Datta Lab at Harvard Medical School.

Latest version is `0.5.0`

***

## Features
Below are the commands/functionality that moseq2-extract currently affords. 
They are accessible via CLI or Jupyter Notebook in [moseq2-app](https://github.com/dattalab/moseq2-app/tree/release).
```bash
Usage: moseq2-extract [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.  [default: False]

Commands:
  convert-raw-to-avi  Converts/Compresses a raw depth file into an avi file...
  copy-slice          Copies a segment of an input depth recording into a...
  download-flip-file  Downloads Flip-correction model that helps with...
  extract             Processes raw input depth recordings to output a...
  find-roi            Finds the ROI and background distance to subtract
                      from...
  generate-config     Generates a configuration file that holds editable...
  version             Print version number
```

Run any command with the `--help` flag to display all available options and their descriptions.

***

## Documentation

All documentation regarding moseq2-extract can be found in the `Documentation.pdf` file in the root directory,
an HTML ReadTheDocs page can be generated via running the `make html` in the `docs/` directory.

For information on getting started, check out the [MoSeq Roadmap](https://github.com/dattalab/moseq2-docs/wiki).
***

## Examples

### Example `find-roi` Outputs

#### Round Arena ROI and Background

<img src="https://drive.google.com/uc?export=view&id=1v8GAgWJu-Gcvf9OhkoHX6G2SXmH5a4D_"></li><br><br>

#### Y-Maze Arena (from Kinect v2 Camera)

<img src="https://drive.google.com/uc?export=view&id=1w21Di6TsRg-Hgbd2PCwIU_kyrvGuajar" width=350 height=350>

#### Convex-shaped Bucket (\_/) (from Azure Camera)
Dilated Background and ROI respectively.

<img src="https://drive.google.com/uc?export=view&id=1HObbzfZF1OXD0h_HBEF9G6-2ExEoc0HE">
<img src="https://drive.google.com/uc?export=view&id=1A_lQ03tiUWDiqMW3Ov7TDyV-_lH3km40">

#### Rectangular Arena (from RealSense)
ROI and Weighted Background Image respectively.

<img src="https://drive.google.com/uc?export=view&id=1Emx-Vlsxp7kM1QVIZ01Wi7bgHHAmx-TT">

### Example Extraction

<img src="https://drive.google.com/uc?export=view&id=1qg_twPau5g0hWpvnGUQzl_7_XdSSsL1c" width=350 height=350>

***

## Contributing

If you would like to contribute, fork the repository and issue a pull request.  
