import pydicom
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# TASK: In the same folder as this .py file you will find a DICOM file
# that is called instance.dcm. Your goal is to save it as a png file. 
# You must apply a WW/WC transform in order to map values to grayscale.

# 1. Use pydicom's dcmread method to load the image

# <YOUR CODE HERE>
dcm = pydicom.dcmread('images/dicom_instance.dcm')
print(type(dcm), type(dcm.pixel_array))
print(f'Modality {dcm.Modality}')

# 2. Apply WW/WC transform. Keep in mind that pixels are available as 
# NumPy array by accessing .pixel_array field.
# You might want to display the image to check that it comes out right.
# You can use matplotlib's plt.imshow method to display image as 
# grayscale data like so: plt.imshow(pixels, cmap="gray")

# <YOUR CODE HERE>
pixels = np.copy(dcm.pixel_array)
print(f"[Initial]\t min: {np.min(pixels)}, max: {np.max(pixels)}")

hu_min = 500 - 1000
hu_max = 500 + 1000

pixels[np.where(pixels < hu_min)] = hu_min
pixels[np.where(pixels > hu_max)] = hu_max
print(f"[Window]\t min: {np.min(pixels)}, max: {np.max(pixels)}")

pixels = (pixels - hu_min)/(hu_max-hu_min)
print(f"[Normalize]\t min: {np.min(pixels)}, max: {np.max(pixels)}")

plt.imshow(pixels, cmap="gray")
plt.show()

# 3. You can use PIL's Image.fromarray and .save methods to save a NumPy array as png
# Don't forget that when you are using "L" mode, PIL is expecting your data to 
# be of type uint8 (and in the range of 0-255)
pixels = (pixels * 255).astype(np.uint8)
print(f"[Grayscale]\t min: {np.min(pixels)}, max: {np.max(pixels)}")

pixels = Image.fromarray(pixels, mode="L")
pixels.save('out.png')

