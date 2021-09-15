# DeepVO
The repository is an unofficial PyTorch implementation of DeepVO. Here is the [link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7989236) to the ICRA 2017 paper. The codes are mainly based on another unofficial [implementation](https://github.com/ChiWeiHsiao/DeepVO-pytorch) of DeepVO but with several modifications and extentions. 

## Dataset
Please download the KITTI Odometry dataset (color, 65GB) following this [link](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and place it under the `dataset` folder. The data path should look like `dataset/sequence/00`. The ground truth poses are provided under the `dataset/poses` folder.

## Preprocessing
Run `preprocessing.py` to get the ground truth for relative poses. The 6DoF relative poses are represented as rotations (roll, pitch, yaw) and translations (x, y, z). However, different from the original [repository](https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/preprocess.py), we first compute the 4x4 transformation matrix between adjacent frames and then convert it into rotations and translations. 

For example, we have two 4x4 transformation matrices M1 and M2 that represent the global poses (relative to the starting point) at time 1 and 2. Then, the relative pose transformation matrix M' between time 1 and 2 can be calculated by M' = inv(M1) * (M2). Then, we can extract R' and t' from M'. t' is the relative translation and the relative rotations can be extracted from R' using function `euler_from_matrix` in `helper.py`.

## Pre-trained FlowNet model
We refer people to the original [github](https://github.com/ChiWeiHsiao/DeepVO-pytorch) to download the pre-trained FlowNet. The pretrained weights should be placed under `models` folder. 

## Train the DeepVO model
Run `main.py` to train the DeepVO model. The model will be saved at `experiments/experiment_name/models` and the log file will be stored under `experiments/experiment_name/record`. The hyper-parameters are included in `params.py` and instructions are provided inside. 

Highlighted modifications and extensions:
  - `get_partition_data_info` in `data_helper.py` is modified to guarantee that the frames in an image sequence are consecutive
  - Add horiziontal flipping as a method for data augmentation

## Test the DeepVO model
Run `test.py` to generate estimations under `experiments/experiment_name/results`. Different from the original [github repository](https://github.com/ChiWeiHsiao/DeepVO-pytorch), the entire test video is fed to the LSTM and the relative poses are generated sequentially. 

To evaluate the test results, run `evaluation.py` to generate visualizations under `experiments/experiment_name/results`. The evaluation code is borrowed from [link](https://github.com/LeoQLi/KITTI_odometry_evaluation_tool). 

## Results 

