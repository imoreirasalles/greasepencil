import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import match_template, peak_local_max
from skimage.morphology import extrema
from skimage.measure import label
from skimage.segmentation import watershed
from skimage import exposure
from skimage import img_as_ubyte, io, exposure, color

source = input("Em qual pasta estão as imagens matriz?")
destination = input("Em qual pasta deseja salvar as imagens individualizadas?")

def greasepencil(input_image):

    os.chdir(f"{source}")

    # read image and determine template width
    color_image = io.imread(input_image)
    image = img_as_ubyte(io.imread(input_image, as_gray=True))
    tmp_w = round(0.9*min(image.shape)/4)

    # stablish relative measures
    frame_left = round(tmp_w * 0.15)
    frame_right = round(tmp_w * 0.85)
    frame_top = round(tmp_w * 0.27)
    frame_bottom = round(tmp_w * 0.73)

    # build slide template
    rel_template = np.full((tmp_w, tmp_w), 0.07)
    rel_template[frame_top:frame_bottom, frame_left:frame_right] = 0.5
    rel_template = np.pad(rel_template, (20,), constant_values=1)
    rel_template += 0.1 * np.random.random(rel_template.shape)

    v_template = np.rot90(rel_template)

    # template matching
    h_result = match_template(color_image[:,:,0], rel_template, pad_input=True)
    v_result = match_template(color_image[:,:,0], v_template, pad_input=True)

    # find and sort matching peaks (top left to bottom right)
    h_coordinates = peak_local_max(h_result, min_distance=int(tmp_w*0.8), threshold_abs=0.35, exclude_border=False)
    coordinates_list = [[coor[0], coor[1]] for coor in h_coordinates]
    coordinates_list.sort(key= lambda item: (item[0] * 4 + item[1]) / 5)

    
    # determine picture orientation and save file
    counter = 1

    for point in (coordinates_list):
        r, c = point
        name, ext = str.split(input_image, ".")
        if v_result[r, c] > h_result[r, c]:
            io.imsave(
                f"{destination}/{name}-{counter:02}.jpg", 
                color_image
                [r - round(tmp_w * 0.35) : r + round(tmp_w * 0.35), 
                c - round(tmp_w * 0.23) : c + round(tmp_w * 0.23)], 
                quality=100
            )
        else:       
            io.imsave(
                f"{destination}/{name}-{counter:02}.jpg",
                color_image
                [r - round(tmp_w * 0.23) : r + round(tmp_w * 0.23), 
                c - round(tmp_w * 0.35) : c + round(tmp_w * 0.35)], 
                quality=100
            )
        
        counter += 1


# create list of files to be processed
extensions = (".jpg", ".jpeg")
image_files = [
    image_file
    for image_file in os.listdir(f"{source}")
    if str(image_file).endswith(tuple(extensions))
]

# call function
for i in image_files:
    greasepencil(i)

# final feedback
print(f"Você forneceu {len(image_files)} imagens e gerou {len(os.listdir(destination))} arquivos!")