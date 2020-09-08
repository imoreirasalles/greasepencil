# Greasepencil

Greasepencil is an under development Python open source software intended to optimize 35mm and medium format digitization workflows. As of now it takes input images containing

## Getting started

Download or clone the repository and install the requirements.

##  Usage

```$ python modules/slides.py -h```

```$ python modules/slides.py /path/to/input/folder /path/to/output/folder```

### CLI args

-tmp_w, --template_width

Slide size to look for. If it is constant across all files, a quick inspection can greatly shorten the processing time.
___
-maxtrim, --maximum_trim (default=30)

Maximum number of dark pixels to crop from output image border. This makes files look nicer but might eat in dark images.
___

-ncol, --number_of_columns (default=4)

How many columns of slides there are in the image. Defaults to a standard vertical printfile.