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


# Write all hyperparameters to record_path
mode = 'a' if par.resume else 'w'
with open(par.record_path, mode) as f:
    f.write('\n' + '=' * 50 + '\n')
    f.write('\n'.join("%s: %s" % item for item in vars(par).items()))
    f.write('\n' + '=' * 50 + '\n')

# Create the training set and test set
train_dataset = ImageSequenceDataset(par.train_video, par.seq_len, drop_last=True, overlap=par.overlap)
train_dl = DataLoader(train_dataset, batch_size=par.batch_size, shuffle=True, num_workers=par.n_processors, pin_memory=par.pin_mem)

valid_dataset = ImageSequenceDataset(par.valid_video, par.seq_len, drop_last=True, overlap=1)
valid_dl = DataLoader(valid_dataset, batch_size=par.batch_size, shuffle=True, num_workers=par.n_processors, pin_memory=par.pin_mem)

print('Number of samples in training dataset: ', len(train_dataset))
print('Number of samples in validation dataset: ', len(valid_dataset))
print('=' * 50)

# Create the model
M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
M_deepvo.load_Flownet()
M_deepvo = M_deepvo.to(par.device)

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
    print('Epoch ' + str(ep) + ' starts')

    loss_mean_train, loss_ang_mean_train, loss_trans_mean_train = M_deepvo.train_net(train_dl, optimizer)
    loss_mean_valid, loss_ang_mean_valid, loss_trans_mean_valid = M_deepvo.valid_net(valid_dl)
    scheduler.step()
    
    message = f'Epoch {ep + 1}\ntrain loss mean: {loss_mean_train}, train ang loss mean: {loss_ang_mean_train}, train trans loss mean: {loss_trans_mean_train}, time: {time.time()-st_t} sec'
    print(message)
    f = open(par.record_path, 'a')
    f.write(message+'\n')
    
    message = f'valid loss mean: {loss_mean_valid}, valid ang loss mean: {loss_ang_mean_valid}, valid trans loss mean: {loss_trans_mean_valid}'
    print(message)
    f = open(par.record_path, 'a')
    f.write(message+'\n')

    # Save model
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

    print(f'Save model at ep {ep+1}')
    torch.save(M_deepvo.state_dict(), par.save_model_path + '.latest')
    torch.save(optimizer.state_dict(), par.save_optimzer_path + '.latest')

