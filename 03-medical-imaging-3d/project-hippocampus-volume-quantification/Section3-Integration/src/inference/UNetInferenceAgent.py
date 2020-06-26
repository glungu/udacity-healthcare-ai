"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        # raise NotImplementedError
        
        patch_size = 64
        volume = med_reshape(volume, new_shape=(volume.shape[0], patch_size, patch_size))
        
        sh = volume.shape
        data = np.zeros((sh[0], 1, sh[1], sh[2]))
        data[:, 0, :, :] = volume / 255.
        data = torch.from_numpy(data).float().cpu()

        # print(f'Inference, data: {data.shape}')
        prediction = self.model(data)
        pred = prediction.cpu().detach().numpy()
        # print(f'Inference, pred: {prediction.shape}, numpy: {pred.shape}')
        result = pred.argmax(axis=1)

        return result 

        
        

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>
        
        # create a single batch from the volume
        sh = volume.shape
        data = np.zeros((sh[0], 1, sh[1], sh[2]))
        data[:, 0, :, :] = volume / 255.
        data = torch.from_numpy(data).float().cuda()

        # print(f'Inference, data: {data.shape}')
        prediction = self.model(data)
        pred = prediction.cpu().detach().numpy()
        # print(f'Inference, pred: {prediction.shape}, numpy: {pred.shape}')
        result = pred.argmax(axis=1)

        return result 
