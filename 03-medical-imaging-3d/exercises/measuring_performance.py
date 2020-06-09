# In this exercise you will compute some of the performance metrics we have discussed
# Specifically, you will compute Sensitivity and Dice scores 
# between the ground truth for the volume from the previous lesson and the segmentation
# that your network had created. 
# Alternatively, there is a second, auto-generated segmentation available for you to 
# compare against
#
# Below we provide some starter code and put comments prepended with TASK to indicate 
# places where you need to fill in

import numpy as np
import nibabel as nib

if __name__ == "__main__":

    # TASK: Load segmentation masks from the nifti files in the folder data,
    # plus your own mask if you have created one in the previous exercise
    # spleen1_label_gt is the ground truth mask
    # spleen1_label_auto is the auto-generated one
    
    # <YOUR CODE HERE>
    
    data_auto = nib.load("data/spleen1_label_auto.nii.gz").get_fdata()
    print(f'Auto: {type(data_auto)}, {data_auto.shape}')
    data_gt = nib.load("data/spleen1_label_gt.nii.gz").get_fdata()
    print(f'Auto: {type(data_gt)}, {data_gt.shape}')

    # TASK: Now, implement two similarity metrics - sensitivity (assuming that gt is the True mask)
    # and Dice Similarity Coefficient. 
    # Hint: the formal measure of "set cardinality" that is featured in DSC definition could
    # be computed as simply the volume of your 3D object
    # <YOUR CODE HERE> 
    sensitivity = np.logical_and(data_auto, data_gt).sum()/data_gt.sum()
    print(f'Sensitivity: {sensitivity}')
    
    dice = 2.0*np.logical_and(data_auto, data_gt).sum()/(data_auto.sum() + data_gt.sum())
    print(f'Dice: {dice}')
    