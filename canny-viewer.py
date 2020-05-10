import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import io
from skimage import exposure
from skimage import transform
from skimage import filters
from skimage import feature
from skimage.morphology import closing, square, remove_small_objects
from skimage import img_as_ubyte
import skimage.viewer


image = io.imread('/Users/martimpassos/dev/greasepencil/_input/P012CR0475F1-2.jpg', as_gray=True)
viewer = skimage.viewer.ImageViewer(image)

# Create the plugin and give it a name
canny_plugin = skimage.viewer.plugins.Plugin(image_filter=skimage.feature.canny)
canny_plugin.name = "Canny Filter Plugin"

# Add sliders for the parameters
canny_plugin += skimage.viewer.widgets.Slider(
    name="sigma", low=0.0, high=7.0, value=2.0
)
canny_plugin += skimage.viewer.widgets.Slider(
    name="low_threshold", low=0.0, high=1.0, value=0.1
)
canny_plugin += skimage.viewer.widgets.Slider(
    name="high_threshold", low=0.0, high=1.0, value=0.2
)

# add the plugin to the viewer and show the window
viewer += canny_plugin
viewer.show()

#image[image > 0.7] = 255
#image[image <= 0.7] = 0

#gamma_corrected = exposure.adjust_gamma(image, 1)
#closed = closing(image, square(5))
#edges = feature.canny(closed, sigma=1.5)

'''fig, (ax1) = plt.subplots(sharex=True, sharey=True)

ax1.imshow(edges)
ax1.axis('off')

fig.tight_layout()

plt.show()'''

#io.imsave('001FE001_edges.jpg', img_as_ubyte(edges))