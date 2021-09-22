# predicted as a batch
from params import par
from model import DeepVO
import numpy as np
from PIL import Image
import glob
import os
import torch
from utils.data import get_data_info
from utils.util import pose_to_SE3
import torchvision.transforms.functional as TF


def test(path_list):
    # Path
    load_model_path = par.load_model_path  # choose the model you want to load
    save_dir = '{}/results/'.format(par.save_path)  # directory to save prediction answer
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    device = torch.device('cuda:1')

    # Load model
    M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        M_deepvo = M_deepvo.to(par.device)
        M_deepvo.load_state_dict(torch.load(load_model_path))
    else:
        M_deepvo.load_state_dict(torch.load(load_model_path, map_location={'cpu'}))
    print('Load model from: ', load_model_path)
    

    M_deepvo.eval()
    for test_video in path_list:
        data = get_data_info(folder_list=[test_video], seq_len=10, drop_last=False, overlap=1)
        image_arr = data['image_path'] # image paths
        groundtruth_arr = data['pose']
        
        prev = None
        answer = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], ]
        ang_err_list = []
        trans_err_list = []
        for i in range(len(data['image_path'])):
            
            print('{} / {}'.format(i, len(data['image_path'])), end='\r', flush=True)
            # Load the test images
            image_path_sequence = image_arr[i]
            image_sequence = []
            for img_path in image_path_sequence:
                img_as_img = Image.open(img_path)
                img_as_img = TF.resize(img_as_img, size=(par.img_h, par.img_w))
                img_as_tensor = TF.to_tensor(img_as_img)-0.5
                img_as_tensor = img_as_tensor.unsqueeze(0)
                image_sequence.append(img_as_tensor)
            image_sequence = torch.cat(image_sequence, 0).to(par.device)

            gt_sequence = groundtruth_arr[i][:, :6]

            with torch.no_grad():
                x_in = image_sequence.unsqueeze(0)
                angle, trans, hc = M_deepvo.forward(x_in, prev=prev)
                prev = hc

            angle = angle.squeeze().detach().cpu().numpy()
            trans = trans.squeeze().detach().cpu().numpy()
            pose_pred = np.hstack((angle, trans))

            # Record the estimation error
            ang_err_list.append(np.mean((gt_sequence[:, :3] - angle)**2))
            trans_err_list.append(np.mean((gt_sequence[:, 3:] - trans)**2))
            
            # Accumulate the relative poses
            for index in range(angle.shape[0]):
                poses_pre = answer[-1]
                poses_pre = np.array(poses_pre).reshape(3, 4)
                R_pre = poses_pre[:, :3]
                t_pre = poses_pre[:, 3]

                pose_rel = pose_pred[index, :]                
                Rt_rel = pose_to_SE3(pose_rel)
                R_rel = Rt_rel[:, :3]
                t_rel = Rt_rel[:, 3]

                R = R_pre @ R_rel
                t = R_pre.dot(t_rel) + t_pre

                pose = np.concatenate((R, t.reshape(3, 1)), 1).flatten().tolist()
                answer.append(pose)


        print('len(answer): ', len(answer))
        print('expect len: ', len(glob.glob('{}{}/*.png'.format(par.image_dir, test_video))))

        
        ang_err_m = np.mean(ang_err_list)
        trans_err_m = np.mean(trans_err_list)

        print(f'Average angle loss: {ang_err_m}')
        print(f'Average translation loss: {trans_err_m}')


        # Save answer
        with open('{}/{}_pred.txt'.format(save_dir, test_video), 'w') as f:
            for pose in answer:
                if type(pose) == list:
                    f.write(' '.join([str(r) for r in pose]))
                else:
                    f.write(str(pose))
                f.write('\n')


if __name__ == '__main__':
    test(par.test_video)
    #test(par.train_video)