import os
import torch

class Parameters():
    def __init__(self):

        self.n_processors = 8
        self.device = device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

        # Path to the dataset. Please modify this before running
        self.data_dir = '/home/mingyuy/DeepVO/dataset'
        self.image_dir = self.data_dir + '/sequences/'
        self.pose_dir = self.data_dir + '/poses/'
        
        # List of train paths, valid paths and test paths
        self.train_video = ['00', '02', '08', '09']
        self.valid_video = ['03', '05']
        self.test_video = ['03', '04', '05', '06', '07', '10']

        self.seq_len = 7           # Image sequence length
        self.overlap = 1           # overlap between adjacent sampled image sequences

        # Data Preprocessing
        self.img_w = 640   # 1280 in the original paper
        self.img_h = 192   # 384 in the original paper

        # Data Augmentation (horizontal flip)
        self.is_hflip = False
        
        # Neural network settings
        self.rnn_hidden_size = 1000
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.2
        self.rnn_dropout_between = 0.2 
        self.batch_norm = True

        # Training settings
        self.epochs = 130
        self.batch_size = 12
        self.pin_mem = True
        self.optim_lr = 1e-3
        self.optim_decay = 5e-6
        self.optim_lr_decay_factor = 0.1
        self.optim_lr_step = 60

        # Pretrain, Resume training
        self.pretrained_flownet = './flownets_bn_EPE2.459.pth.tar'
        self.resume = False
        self.resume_t_or_v = '.latest'
        
        # Paths to save and load the model
        self.experiment_name = 'experiment_name'
        self.save_path = 'experiments/{}'.format(self.experiment_name)

        self.name = 't{}_v{}_im{}x{}_s{}_b{}'.format(''.join(self.train_video), ''.join(self.test_video), self.img_h, self.img_w, self.seq_len, self.batch_size)
        self.name += '_flip' if self.is_hflip else ''

        self.load_model_path = '{}/models/{}.model{}'.format(self.save_path, self.name, self.resume_t_or_v)
        self.load_optimizer_path = '{}/models/{}.optimizer{}'.format(self.save_path, self.name, self.resume_t_or_v)
        self.record_path = '{}/records/{}.txt'.format(self.save_path, self.name)
        self.save_model_path = '{}/models/{}.model'.format(self.save_path, self.name)
        self.save_optimzer_path = '{}/models/{}.optimizer'.format(self.save_path, self.name)

        if not os.path.isdir(os.path.dirname(self.record_path)):
            os.makedirs(os.path.dirname(self.record_path))
        if not os.path.isdir(os.path.dirname(self.save_model_path)):
            os.makedirs(os.path.dirname(self.save_model_path))
        if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
            os.makedirs(os.path.dirname(self.save_optimzer_path))


par = Parameters()
