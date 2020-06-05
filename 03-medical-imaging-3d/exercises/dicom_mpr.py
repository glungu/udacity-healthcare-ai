#%%
import pydicom
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# TASK: In the same folder as this .py file you will find a DICOM folder
# with a volume in it. You can assume that all files belong to the same series.
# Your task is to visualize any three slices across the three 
# cardinal planes from this volume: Axial, Coronal and Sagittal. You can visualize 
# by saving them as png, similar to the previous exercise. You can visualize slice at 
# any level, but I suggest that you visualize slices located in the middle of the volume.

# 
# Bonus points: 
# 1) What is the modality that you are dealing with here?
# 2) Try to figure out which axis corresponds to what plane by searching online.
# You should have a good guess of what anatomy you are looking at if you visualize the middle slice
# 3) Try plotting the slices in non-primary planes with proper aspect ratio
#
# Hints:
# - You may want to rescale the output because your voxels are non-square. 
# - Don't forget that you need to order your slices properly. Filename 
# may not be the best indicator of the slice order. 
# If you're confused, try looking up the first value of ImagePositionPatient
# - Don't forget the windowing. A good initial guess would be scaling all
# image values down to [0..1] range when saving. Pillow deals with such well

# %% 
# Load the volume into array of slices
path = f"volume"
slices = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]

# <YOUR CODE HERE>
print(f'Num slices: {len(slices)}')
print(slices[0])
print(f"Modality: {slices[0].Modality}")
print(f"Slice Thickness: {slices[0].SliceThickness}mm")
print(f"Pixel Spacing: {slices[0].PixelSpacing}")
print(f"Image Orientation Patient: {slices[0].ImageOrientationPatient}")

cosine_x = np.array(slices[0].ImageOrientationPatient[3:], dtype=float)
index_x = np.argwhere(cosine_x > 0.9).flatten()[0]

cosine_y = np.array(slices[0].ImageOrientationPatient[:3], dtype=float)
index_y = np.argwhere(cosine_y > 0.9).flatten()[0]

print(f'Orientation: X-axis index: {index_x}, Y-axis index: {index_y}')


# for s in slices:
#     print(f'ImagePositionPatient: {s.ImagePositionPatient}')

slices = sorted(slices, key=lambda s: s.ImagePositionPatient[0])

volume = np.stack([s.pixel_array for s in slices])
print(f'Volume shape: {volume.shape}, type: {volume.dtype}')

def dcm_to_png(dcm_pixel_array, resize, outfile):
    pixels = np.copy(dcm_pixel_array)
    hu_min = -1000
    hu_max = +1000

    # window by intensity
    pixels[np.where(pixels < hu_min)] = hu_min
    pixels[np.where(pixels > hu_max)] = hu_max

    # normalize   
    pixels = (pixels - hu_min)/(hu_max-hu_min)

    # convert to 8-bit grascale (0..255)
    pixels = (pixels * 255).astype(np.uint8)

    # create grayscale image
    image = Image.fromarray(pixels, mode="L")

    # resize 
    if resize is not None:
        newsize = (int(image.size[0]*resize[0]), int(image.size[1]*resize[1]))
        image = image.resize(newsize)

    # save
    image.save(outfile)
    print(f'File {outfile} saved')


axial_ind = 17
axial = volume[axial_ind,:,:]
plt.imshow(axial, cmap = "gray")
plt.title(f'Axial {axial_ind}/{volume.shape[0]}')
plt.show()
dcm_to_png(axial, None, 'images/out_axial.png')


coronal_ind = 70
coronal = volume[:,coronal_ind,:]
ratio_zy = slices[0].SliceThickness/slices[0].PixelSpacing[1]
plt.imshow(coronal, cmap="gray", aspect = ratio_zy)
plt.title(f'Coronal {coronal_ind}/{volume.shape[1]}')
plt.show()
print(f'Aspect ratio: {ratio_zy}')
dcm_to_png(coronal, (1, ratio_zy), 'images/out_coronal.png')


sagittal_ind = 130
sagittal = volume[:,:,sagittal_ind]
ratio_zx = slices[0].SliceThickness/slices[0].PixelSpacing[0]
plt.imshow(sagittal, cmap="gray", aspect = ratio_zx)
plt.title(f'Sagittal {sagittal_ind}/{volume.shape[2]}')
plt.show()
print(f'Aspect ratio: {ratio_zx}')
dcm_to_png(sagittal, (1, ratio_zx), 'images/out_sagittal.png')
