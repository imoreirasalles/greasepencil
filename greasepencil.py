import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import match_template, peak_local_max
from skimage.morphology import extrema
from skimage.measure import label
from skimage.segmentation import watershed
from skimage import exposure, img_as_ubyte, io, color, transform

"""
PAD WITH BLACK TO CENTER TEMPLATE
"""

source = input("Em qual pasta estão as imagens matriz?")
destination = input("Em qual pasta deseja salvar as imagens individualizadas?")
previous_files = len(os.listdir(destination))

def trim_black_border(img,tol=0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    
    col_start = np.clip(mask0.argmax(), 0, 23)
    col_end = np.clip(n-mask0[::-1].argmax(), n-23, n)
    row_start = np.clip(mask1.argmax(), 0, 23)
    row_end = np.clip(m-mask1[::-1].argmax(), m-23, m)

    return img[row_start:row_end,col_start:col_end]

def greasepencil(input_image, ncol=4, tmp_w=None):

    os.chdir(f"{source}")

    # read image and determine template width
    color_image = io.imread(input_image)
    image = img_as_ubyte(io.imread(input_image, as_gray=True))
    m, n = image.shape
    if tmp_w is None:
        tmp_w = round(min(image.shape)/ncol)
        user_tmp_w = False
    # establish relative measures
    

def build_template(tmp_w):
    # build slide template
    frame_left = round(tmp_w * 0.15)
    frame_right = round(tmp_w * 0.85)
    frame_top = round(tmp_w * 0.27)
    frame_bottom = round(tmp_w * 0.73)
    
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
        
        tmp_w = tmp_w * scale
        resized = transform.resize(rel_template, (tmp_w, tmp_w))

        v_template = np.rot90(resized)

        h_result = match_template(color_image[:,:,2], resized, pad_input=True)
        v_result = match_template(color_image[:,:,2], v_template, pad_input=True)

        h_coordinates = peak_local_max(h_result, min_distance=int(tmp_w*0.8), threshold_abs=0.35, exclude_border=False)
        coordinates_list = [[coor[0], coor[1]] for coor in h_coordinates]

        if user_tmp_w:
            break

        peaks = [h_result[r, c] for r, c in h_coordinates]
        
        if peaks:
            avg = sum(peaks)/len(peaks)
        else:
            continue
        
        if found is None or avg > found:
            found = avg
            widths.append(tmp_w)
            coors.append(coordinates_list)
            h_results.append(h_result)
            v_results.append(v_result)
        else:
            coordinates_list =  coors[-1]
            v_result = v_results[-1]
            h_result = h_results[-1]
            tmp_w = widths[-1]
            break

    # arrange images from top left to bottom right
    coordinates_list.sort(key= lambda item: (item[0] * 4 + item[1]) / 5)

    # determine picture orientation, trim and save file

    detected_frames = []

    for i, point in enumerate(coordinates_list):
        r, c = point
        name, ext = str.split(input_image, ".")
        large = round(tmp_w * 0.34)
        small = round(tmp_w * 0.22)
        filename = f"{name}-{i+1:02}.jpg"
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


        detected_frames.append({i:i, filename:filename, minr:minr, maxr:maxr, minc:minc, maxc:maxc})
    
    return detected_frames


if __name__ == "__main__":

    # create list of files to be processed
    extensions = (".jpg", ".jpeg")
    image_files = [
        image_file
        for image_file in os.listdir(f"{source}")
        if str(image_file).endswith(tuple(extensions))
    ]

    # call function
    for image in image_files:
        try:
            for frame in greasepencil(image):
                picture = image[frame['minr']:frame['maxr'], frame['minc'], frame['maxc']]
                picture = trim_black_border(picture, tol=100)
                io.imsave(os.path.join(destination, picture["filename"]), picture, quality=100)                
        except Exception as e:
            print(str(e))

    # final feedback
    print(
        f"Você forneceu {len(image_files)} imagens e gerou {len(os.listdir(destination))-previous_files} arquivos!")


