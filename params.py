import os


class Parameters():
    def __init__(self):
        self.n_processors = 8
        # Path
        self.data_dir = '/home/mingyuy/DeepVO/dataset'
        self.image_dir = self.data_dir + '/sequences/'
        self.pose_dir = self.data_dir + '/poses/'

        self.train_video = ['00', '02', '08', '09']
        self.valid_video = ['03', '05']
        self.test_video = ['03', '04', '05', '06', '07', '10']

        self.seq_len = 7           # Image sequence length
        self.sample_times = 1      # sample the image sequence from different starting points
        self.overlap = 6           # overlap between adjacent sampled image sequences

        # Data Preprocessing
        self.img_w = 640   # 1280 in the original paper
        self.img_h = 192   # 384 in the original paper

        # Data Augmentation
        self.is_hflip = False  # Whether to apply horizontal flips as data augmentation
        
        self.rnn_hidden_size = 1000
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.2
        self.rnn_dropout_between = 0.2 
        self.batch_norm = True

        # Training
        self.epochs = 100
        self.batch_size = 16
        self.pin_mem = False
        self.optim_lr = 1e-3
        self.optim_decay = 2e-6
        self.optim_lr_decay_factor = 0.1
        self.optim_lr_step = 40

        # Pretrain, Resume training
        self.pretrained_flownet = './flownets_bn_EPE2.459.pth.tar'
        self.resume = False
        self.resume_t_or_v = '.epoch_91'
        
        self.experiment_name = '0918'
        self.save_path = 'experiments/{}'.format(self.experiment_name)

        self.name = 't{}_v{}_im{}x{}_s{}_b{}'.format(''.join(self.train_video), ''.join(self.test_video), self.img_h, self.img_w, self.seq_len, self.batch_size)

        if self.is_hflip:
            self.name += '_flip'

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
