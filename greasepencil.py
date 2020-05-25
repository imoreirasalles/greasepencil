# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import match_template, peak_local_max
from skimage.morphology import extrema
from skimage.measure import label
from skimage.segmentation import watershed
from skimage import exposure, img_as_ubyte, io, color, transform

from datetime import datetime



# Create argument parser
my_parser = argparse.ArgumentParser(description='Detect frames in a sleeve of 35mm slides')

my_parser.add_argument('source',
                       metavar='Source Path',
                       type=str,
                       help='Path to input files')

my_parser.add_argument('destination',
                       metavar='Destination Path',
                       type=str,
                       help='Path to output files')

my_parser.add_argument('--number_of_columns', '-ncol',
                       nargs = '?', default = 4,
                       type=int,
                       help='Number of columns in sleeve. Defaults to 4 (standard vertical printfile)')

my_parser.add_argument('--template_width', '-tmp_w',
                       nargs = '?', default = None,
                       type = int,
                       help = "size (in pixels) of slides to look for. If empty, the script picks best match starting with image width / ncol and reducing 10percent every try.")

args = my_parser.parse_args()

source = args.source
destination = args.destination
ncol = args.number_of_columns
tmp_w = args.template_width




def trim_black_border(img, tol=0, trim_max=None):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img > tol
    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)

    col_start = np.clip(mask0.argmax(), 0, trim_max)
    row_start = np.clip(mask1.argmax(), 0, trim_max)
    if trim_max == None:
        col_end = (n - mask0[::-1].argmax())
        row_end = (m - mask1[::-1].argmax())
    else:    
        col_end = np.clip(n - mask0[::-1].argmax(), n - trim_max, n)
        row_end = np.clip(m - mask1[::-1].argmax(), m - trim_max, m)

    return img[row_start:row_end, col_start:col_end]


def frame_detector(input_image, ncol=4, tmp_w=None):

    # read image and determine template width
    
    print("Opening and preprocessing")
    rgb_img = img_as_ubyte(io.imread(input_image))
    bw_img = img_as_ubyte(io.imread(input_image, as_gray=True))
    small_bw_img = transform.rescale(bw_img, 0.5)
    #small_bw_img = transform.rescale(small_bw_img, 0.5)
    m, n = bw_img.shape
    if not tmp_w:
        tmp_w = round(small_bw_img.shape[1]/ncol)
        user_tmp_w = False
    else:
        tmp_w = round(tmp_w / 2)
        user_tmp_w = True


    # determine frame edges
    frame_left = round(tmp_w * 0.15)
    frame_right = round(tmp_w * 0.85)
    frame_top = round(tmp_w * 0.27)
    frame_bottom = round(tmp_w * 0.73)

    # build slide template
    print("Building template")
    rel_template = np.full((tmp_w, tmp_w), 0.07)
    rel_template[frame_top:frame_bottom, frame_left:frame_right] = 0.5
    rel_template = np.pad(rel_template, (30,), constant_values=1)
    rel_template += 0.1 * np.random.random(rel_template.shape)

    # loop over template sizes and pick best match
    widths = []
    coors = []
    h_results = []
    v_results = []
    found = None

    for scale in np.linspace(0.5, 1.0, 6)[::-1]:

        print("Looking for objects' positions")
        
        tmp_w = int(rel_template.shape[0] * scale)
        resized = transform.resize(rel_template, (tmp_w, tmp_w))
        v_template = np.rot90(resized)

        print("Horizontal matching...")

        h_result = match_template(small_bw_img, resized, pad_input=True)
        #h_result = transform.rescale(h_result, 4.0)
        
        print("Vertical matching...")

        v_result = match_template(small_bw_img, v_template, pad_input=True)
        #v_result = transform.rescale(v_result, 4.0)
    
        #tmp_w = tmp_w * 4
        
        combined_results = h_result + v_result
        #plt.imshow(combined_results)
        #plt.show()

        coordinates = peak_local_max(
            combined_results,
            min_distance=int(tmp_w * 0.8),
            threshold_abs=0.35,
            exclude_border=False,
        )
        coordinates_list = [(coor[0]*2, coor[1]*2) for coor in coordinates]
        
        h_result = transform.rescale(h_result, 2.0)
        #h_result = transform.rescale(h_result, 2.0)
        
        v_result = transform.rescale(v_result, 2.0)
        #v_result = transform.rescale(v_result, 2.0)

        tmp_w = tmp_w * 2
        #tmp_w = tmp_w * 2


        if user_tmp_w == True:
            print("User defined template size, exiting loop")
            break

        peaks = [combined_results[r, c] for r, c in coordinates]

        if peaks:
            avg = sum(peaks) / len(peaks)
        else:
            print("No objects found, trying again")
            continue

        if found is None or avg > found:
            found = avg
            widths.append(tmp_w)
            coors.append(coordinates_list)
            h_results.append(h_result)
            v_results.append(v_result)
            print("Unsatisfactory results, trying again")
            continue
        else:
            coordinates_list = coors[-1]
            v_result = v_results[-1]
            h_result = h_results[-1]
            tmp_w = widths[-1]
            print("Best match found, exiting loop")
            break

    # sort images from top left to bottom right
    coordinates_list.sort(key=lambda item: (item[0] * 4 + item[1]) / 5)

    # determine picture orientation, trim and save file
    for i, point in enumerate(coordinates_list):
        r, c = point
        name, ext = str.split(os.path.split(input_image)[1], ".")
        large = round(tmp_w * 0.34)
        small = round(tmp_w * 0.22)
        filename = f"{name}-{i+1:02}.jpg"

        print("Determining image orientation")

        if v_result[r, c] > h_result[r, c]:
            minr = np.clip(r - large, 0, None)
            maxr = np.clip(r + large, None, m)
            minc = np.clip(c - small, 0, None)
            maxc = np.clip(c + small, None, n)

        else:
            minr = np.clip(r - small, 0, None)
            maxr = np.clip(r + small, None, m)
            minc = np.clip(c - large, 0, None)
            maxc = np.clip(c + large, None, n)

        picture = rgb_img[minr:maxr, minc:maxc]
        picture = trim_black_border(picture, tol=100, trim_max=30)
        print(f"Saving image {i+1}")
        io.imsave(os.path.join(destination, filename), picture, quality=100)
        


# create list of files to be processed
extensions = (".jpg", ".jpeg")
previous_files = len(os.listdir(f"{destination}"))

image_files = [
    os.path.join(source, image_file)
    for image_file in os.listdir(source)
    if str(image_file).endswith(tuple(extensions))
]

# call function
for i in image_files:
    start = datetime.now()
    frame_detector(i, ncol=ncol, tmp_w=tmp_w)
    print(datetime.now() - start)
    #try:
    #    frame_detector(i)
    #except Exception as e:
    #    print(str(e))

# final feedback
print(
    f"Você forneceu {len(image_files)} imagens e gerou {len(os.listdir(destination))-previous_files} arquivos!"
)

