import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import match_template, peak_local_max
from skimage.morphology import extrema
from skimage.measure import label
from skimage.segmentation import watershed
from skimage import exposure, img_as_ubyte, io, exposure, color, transform

source = input("Em qual pasta estão as imagens matriz?")
destination = input("Em qual pasta deseja salvar as imagens individualizadas?")
previous_files = len(os.listdir(destination))

def greasepencil(input_image, ncol=4, tmp_w=None):

    os.chdir(f"{source}")

    # read image and determine template width
    color_image = io.imread(input_image)
    image = img_as_ubyte(io.imread(input_image, as_gray=True))
    if tmp_w is None:
        tmp_w = round(min(image.shape)/ncol)

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

    # loop over template sizes and pick best match
    count = 0
    widths = []
    coors = []
    h_results = []
    v_results = []
    found = None

    for scale in np.linspace(0.5, 1.0, 6)[::-1]:
        tmp_w = int(rel_template.shape[0]*scale)
        widths.append(tmp_w)
        resized = transform.resize(rel_template, (tmp_w, tmp_w))
        v_template = np.rot90(resized)
        h_result = match_template(color_image[:,:,2], resized, pad_input=True)
        v_result = match_template(color_image[:,:,2], v_template, pad_input=True)
        h_coordinates = peak_local_max(h_result, min_distance=int(tmp_w*0.8), threshold_abs=0.35, exclude_border=False)
        coordinates_list = [[coor[0], coor[1]] for coor in h_coordinates]
        coors.append(coordinates_list)
        h_results.append(h_result)
        v_results.append(v_result)
        peaks = []
        for r, c in h_coordinates:
            peaks.append(h_result[r, c])
        avg = sum(peaks)/len(peaks)
        if found is None or avg > found:
            found = avg
            count += 1
        else:
            coordinates_list =  coors[count - 1]
            v_result = v_results[count - 1]
            h_result = h_results[count - 1]
            tmp_w = widths[count - 1]
            break

    # arrange images from top left to bottom right
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
                [r - round(tmp_w * 0.34) : r + round(tmp_w * 0.34), 
                c - round(tmp_w * 0.22) : c + round(tmp_w * 0.22)], 
                quality=100
            )
        else:       
            io.imsave(
                f"{destination}/{name}-{counter:02}.jpg",
                color_image
                [r - round(tmp_w * 0.22) : r + round(tmp_w * 0.22), 
                c - round(tmp_w * 0.34) : c + round(tmp_w * 0.34)], 
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
    try:
        greasepencil(i)
    except IndexError:
        print(f"a imagem {i} não pôde ser processada")
# final feedback
print(
    f"Você forneceu {len(image_files)} imagens e gerou {len(os.listdir(destination))-previous_files} arquivos!"
)