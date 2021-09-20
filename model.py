import torch
import torch.nn as nn
from params import par
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs 
import torch.nn.functional as F

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )

class DeepVO(nn.Module):
    def __init__(self, imsize1, imsize2, batchNorm=True):
        super(DeepVO,self).__init__()
        # CNN
        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.conv3_1 = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=par.conv_dropout[3])
        self.conv4   = conv(self.batchNorm, 256,  512, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        self.conv4_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=par.conv_dropout[5])
        self.conv5   = conv(self.batchNorm, 512,  512, kernel_size=3, stride=2, dropout=par.conv_dropout[6])
        self.conv5_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=par.conv_dropout[7])
        self.conv6   = conv(self.batchNorm, 512,  1024, kernel_size=3, stride=2, dropout=par.conv_dropout[8])
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self.encode_image(__tmp)
        
        # RNN
        self.rnn = nn.LSTM(
                    input_size=int(np.prod(__tmp.size())), 
                    hidden_size=par.rnn_hidden_size, 
                    num_layers=2, 
                    dropout=par.rnn_dropout_between, 
                    batch_first=True)
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        self.linear = nn.Linear(in_features=par.rnn_hidden_size, out_features=6)

        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)  #orthogonal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)  #orthogonal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        


    def forward(self, x, prev=None):
        # x: (batch, seq_len, channel, width, height)
        # stack_image
        x = torch.cat((x[:, :-1], x[:, 1:]), dim=2)
        batch_size = x.size(0)
        seq_len = x.size(1)
        # CNN
        x = x.view(batch_size*seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)

        # RNN
        if prev is None:
            out, hc = self.rnn(x)
        else:
            out, hc = self.rnn(x, prev)

        out = self.rnn_drop_out(out)
        pose = self.linear(out)
        angle = pose[:, :, :3]
        trans = pose[:, :, 3:]

        return angle, trans, hc

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def get_loss(self, x, y, prev=None):

        angle, trans, _ = self.forward(x, prev=prev)        
        # Weighted MSE Loss
        angle_loss = torch.nn.functional.mse_loss(angle, y[:,:,:3])
        translation_loss = torch.nn.functional.mse_loss(trans, y[:,:,3:])
        yaw_loss = torch.nn.functional.mse_loss(angle[:,:,1], y[:,:,1])

        loss = 100 * angle_loss + translation_loss

        return loss, angle_loss, translation_loss, yaw_loss

    def step(self, x, y, optimizer, prev=None):
        optimizer.zero_grad()
        loss, angle_loss, translation_loss, _ = self.get_loss(x, y, prev=prev)
        loss.backward()
        optimizer.step()
        return loss, angle_loss, translation_loss


