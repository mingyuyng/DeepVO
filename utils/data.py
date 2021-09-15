import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset 
from params import par
import torchvision.transforms.functional as TF


# Generate the data frames per path
def get_data_info(folder_list, seq_len, drop_last, overlap):
    X_path, Y = [], []
    
    for folder in folder_list:
        # Load & sort the raw data
        poses = np.load('{}{}.npy'.format(par.pose_dir, folder))  # (n_images, 6)
        fpaths = glob.glob('{}{}/image_2/*.png'.format(par.image_dir, folder))        
        fpaths.sort()
    
        n_frames = len(fpaths)
        start = 0
        while start + seq_len < n_frames:
            x_seg = fpaths[start:start+seq_len]
            X_path.append(x_seg)
            Y.append(poses[start:start+seq_len-1])
            start += seq_len - overlap

        if not drop_last:
            X_path.append(fpaths[start:])
            Y.append(poses[start:])
    
    # Store in a dictionary
    data = {'image_path': X_path, 'pose': Y}
    return data


class ImageSequenceDataset(Dataset):
    def __init__(self, image_folders, seq_len, drop_last, overlap):
        
        # Load the images paths from folders
        self.data_info = get_data_info(image_folders, seq_len, drop_last, overlap) 
        self.image = self.data_info['image_path']  # image paths
        self.groundtruth = self.data_info['pose']
    
    def __getitem__(self, index):
        
        # Have a 50% probability of horizonally flipping the image if enabled 
        flag_hflip = torch.rand(1) > 0.5 if par.is_hflip else False
        
        # Prepare to load the images
        image_path_sequence = self.image[index]
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_img = TF.resize(img_as_img, size=(par.img_h, par.img_w))
            if flag_hflip:
                img_as_img = TF.hflip(img_as_img)
            img_as_tensor = TF.to_tensor(img_as_img) - 0.5 # The FlowNet CNN is pre-trained with [-0.5, 0.5]
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)

        # Prepare the ground truth pose
        gt_sequence = self.groundtruth[index][:, :6]
        gt_sequence = torch.FloatTensor(gt_sequence)

        # If apply horizontal flip, need to modify the ground truth poses as well 
        if flag_hflip:
            for gt_seq in gt_sequence:
                gt_seq[1], gt_seq[2], gt_seq[3] = -gt_seq[1], -gt_seq[2], -gt_seq[3]  # Flip the Pitch, Yaw angle and translation in x-axes 
                
        return (image_sequence, gt_sequence)

    def __len__(self):
        return len(self.data_info['image_path'])


# Example of usage
if __name__ == '__main__':
    df = get_data_info('00', 5)
