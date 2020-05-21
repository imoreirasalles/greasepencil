import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import match_template, peak_local_max
from skimage.morphology import extrema
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.color import rgb2hsv
from skimage import exposure, img_as_ubyte, io, color, transform

from datetime import datetime

source = input("Em qual pasta estão as imagens matriz?")
destination = input("Em qual pasta deseja salvar as imagens individualizadas?")
previous_files = len(os.listdir(destination))


def trim_black_border(img, tol=0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img > tol
    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)

    col_start = np.clip(mask0.argmax(), 0, 23)
    col_end = np.clip(n - mask0[::-1].argmax(), n - 23, n)
    row_start = np.clip(mask1.argmax(), 0, 23)
    row_end = np.clip(m - mask1[::-1].argmax(), m - 23, m)

    return img[row_start:row_end, col_start:col_end]


def frame_detector(input_image, ncol=4, tmp_w=None):

    os.chdir(f"{source}")

    # read image and determine template width
    print("Opening and preprocessing")
    rgb_img = img_as_ubyte(io.imread(input_image))
    bw_img = img_as_ubyte(io.imread(input_image, as_gray=True))
    #hsv_img = img_as_ubyte(rgb2hsv(rgb_img))
    #value_img = hsv_img[:, :, 2]
    m, n = bw_img.shape
    if not tmp_w:
        tmp_w = round(min(bw_img.shape)/ncol)
        user_tmp_w = False
    else:
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
        h_result = match_template(bw_img, resized, pad_input=True)
        print("Vertical matching...")
        v_result = match_template(bw_img, v_template, pad_input=True)

        combined_results = h_result + v_result
        
        coordinates = peak_local_max(
            combined_results,
            min_distance=int(tmp_w * 0.8),
            threshold_abs=0.35,
            exclude_border=False,
        )
        coordinates_list = [[coor[0], coor[1]] for coor in coordinates]


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

    # arrange images from top left to bottom right
    coordinates_list.sort(key=lambda item: (item[0] * 4 + item[1]) / 5)

    # determine picture orientation, trim and save file
    for i, point in enumerate(coordinates_list):
        r, c = point
        name, ext = str.split(input_image, ".")
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
        picture = trim_black_border(picture, tol=100)
        print(f"Saving image {i+1}")
        io.imsave(os.path.join(destination, filename), picture, quality=100)



# create list of files to be processed
extensions = (".jpg", ".jpeg")
image_files = [
    image_file
    for image_file in os.listdir(f"{source}")
    if str(image_file).endswith(tuple(extensions))
]

# call function
for i in image_files:
    try:
        frame_detector(i, tmp_w=930)
    except Exception as e:
        print(str(e))

# final feedback
print(
    f"Você forneceu {len(image_files)} imagens e gerou {len(os.listdir(destination))-previous_files} arquivos!"
)

