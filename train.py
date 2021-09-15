import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from params import par
from model import DeepVO
from utils.data import ImageSequenceDataset
import math
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(0)
np.random.seed(0)

# Setup the device
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

# Write all hyperparameters to record_path
mode = 'a' if par.resume else 'w'
with open(par.record_path, mode) as f:
    f.write('\n' + '=' * 50 + '\n')
    f.write('\n'.join("%s: %s" % item for item in vars(par).items()))
    f.write('\n' + '=' * 50 + '\n')


train_dataset = ImageSequenceDataset(par.train_video, par.seq_len, drop_last=True, overlap=par.overlap)
train_dl = DataLoader(train_dataset, batch_size=par.batch_size, shuffle=True, num_workers=par.n_processors, pin_memory=par.pin_mem)

valid_dataset = ImageSequenceDataset(par.valid_video, par.seq_len, drop_last=True, overlap=1)
valid_dl = DataLoader(valid_dataset, batch_size=par.batch_size, shuffle=True, num_workers=par.n_processors, pin_memory=par.pin_mem)

print('Number of samples in training dataset: ', len(train_dataset))
print('Number of samples in validation dataset: ', len(valid_dataset))
print('=' * 50)

# Model
M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)

# Load the pre-trained FlowNet
pretrained_w = torch.load(par.pretrained_flownet, map_location='cpu')

# Load FlowNet weights pretrained with FlyingChairs
# NOTE: the pretrained model assumes image rgb values in range [-0.5, 0.5]
if par.pretrained_flownet and not par.resume:
    # Use only conv-layer-part of FlowNet as CNN for DeepVO
    model_dict = M_deepvo.state_dict()
    update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
    model_dict.update(update_dict)
    M_deepvo.load_state_dict(model_dict)

del pretrained_w
M_deepvo = M_deepvo.to(device)

optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=par.optim_lr, betas=(0.9, 0.999), weight_decay=par.optim_decay)
scheduler = StepLR(optimizer, gamma=par.optim_lr_decay_factor, step_size=par.optim_lr_step)

# Load trained DeepVO model and optimizer
if par.resume:
    M_deepvo.load_state_dict(torch.load(par.load_model_path))
    optimizer.load_state_dict(torch.load(par.load_optimizer_path))
    print('Load model from: ', par.load_model_path)
    print('Load optimizer from: ', par.load_optimizer_path)


# Train
print('Record loss in: ', par.record_path)
min_loss_train = 1e10
min_loss_valid = 1e10

for ep in range(par.epochs):

    st_t = time.time()
    print('=' * 50)
    # Train
    M_deepvo.train()
    loss_ang_mean_train = 0
    loss_trans_mean_train = 0
    iter_num = 0

    for t_x, t_y in train_dl:

        t_x = t_x.to(device)
        t_y = t_y.to(device)
        
        loss, angle_loss, translation_loss = M_deepvo.step(t_x, t_y, optimizer)
        loss = loss.data.cpu().numpy()
        angle_loss = angle_loss.data.cpu().numpy()
        translation_loss = translation_loss.data.cpu().numpy()

        loss_ang_mean_train += float(angle_loss) * 100 
        loss_trans_mean_train += float(translation_loss)

        iter_num += 1
        if iter_num % 25 == 0:
            message = f'Epoch:{ep}, Iteration: {iter_num}, Loss: {loss:.3f}, angle: {100*angle_loss:.4f}, trans: {translation_loss:.3f}, Train take {time.time()-st_t:.1f} sec'
            f = open(par.record_path, 'a')
            f.write(message+'\n') 
            print(message)
    
    scheduler.step()
    loss_ang_mean_train /= len(train_dl)
    loss_trans_mean_train /= len(train_dl)
    loss_mean_train = loss_ang_mean_train + loss_trans_mean_train


    # Validation
    st_t = time.time()
    M_deepvo.eval()
    loss_ang_mean_valid = 0
    loss_trans_mean_valid = 0

    for v_x, v_y in valid_dl:

        v_x = v_x.to(device)
        v_y = v_y.to(device)

        loss, angle_loss, translation_loss = M_deepvo.get_loss(v_x, v_y)
        loss = loss.data.cpu().numpy()
        angle_loss = angle_loss.data.cpu().numpy()
        translation_loss = translation_loss.data.cpu().numpy()

        loss_ang_mean_valid += float(angle_loss) * 100 
        loss_trans_mean_valid += float(translation_loss)

    print('Valid take {:.1f} sec'.format(time.time() - st_t))
    loss_ang_mean_valid /= len(valid_dl)
    loss_trans_mean_valid /= len(valid_dl)
    loss_mean_valid = loss_ang_mean_valid + loss_trans_mean_valid

    message = f'Epoch {ep + 1}\ntrain loss mean: {loss_mean_train}, train ang loss mean: {loss_ang_mean_train}, train trans loss mean: {loss_trans_mean_train}'
    print(message)
    f = open(par.record_path, 'a')
    f.write(message+'\n')
    
    message = f'valid loss mean: {loss_mean_valid}, valid ang loss mean: {loss_ang_mean_valid}, valid trans loss mean: {loss_trans_mean_valid}'
    print(message)
    f = open(par.record_path, 'a')
    f.write(message+'\n')

    # Save model
    # save if the valid loss decrease
    check_interval = 1
    if loss_mean_valid < min_loss_valid and ep % check_interval == 0:
        min_loss_valid = loss_mean_valid
        print('Save model at ep {}, mean of valid loss: {}'.format(ep + 1, loss_mean_valid))
        torch.save(M_deepvo.state_dict(), par.save_model_path + '.valid')
        torch.save(optimizer.state_dict(), par.save_optimzer_path + '.valid')

    if loss_mean_train < min_loss_train and ep % check_interval == 0:
        min_loss_train = loss_mean_train
        print('Save model at ep {}, mean of train loss: {}'.format(ep + 1, loss_mean_train))
        torch.save(M_deepvo.state_dict(), par.save_model_path + '.train')
        torch.save(optimizer.state_dict(), par.save_optimzer_path + '.train')

    if (ep+1) % 5 == 0:
        print(f'Save model at ep {ep+1}')
        torch.save(M_deepvo.state_dict(), par.save_model_path + '.epoch_' + str(ep+1))
        torch.save(optimizer.state_dict(), par.save_optimzer_path + '.epoch_' + str(ep+1))

