import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from data_util import read_sequential_target
from config import TrainConfig

# Evaluate the performance of action generation in terms of NRMSE loss between predicted and ground truth joint values
def evaluate():
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    B_fw, B_bw, B_bin, B_len, filenames = read_sequential_target(train_conf.B_dir, True)
    B_fw_u, B_bw_u, B_bin_u, B_len_u, filenames_u = read_sequential_target(train_conf.B_dir_test, True)
    min = B_fw.min()
    max = B_fw.max()
    min_u = B_fw_u.min()
    max_u = B_fw_u.max()
    mean_u = B_fw_u.mean()
    #B_fw = 2 * ((B_fw - B_fw.min())/(B_fw.max()-B_fw.min())) - 1
    #B_fw_u = 2 * ((B_fw_u - B_fw_u.min()) / (B_fw_u.max() - B_fw_u.min())) - 1
    B_fw = B_fw.transpose((1,0,2))
    B_fw_u = B_fw_u.transpose((1,0,2))
    B_bw = B_bw.transpose((1,0,2))
    B_bw_u = B_bw_u.transpose((1,0,2))
    B_bin = B_bin.transpose((1,0,2))
    B_bin_u = B_bin_u.transpose((1,0,2))
    predict_train, _, predtrain_bin, predtrain_len, _ = read_sequential_target('../train/inference/prediction/behavior_train/', True)
    predict_test, _, predtest_bin, predtest_len, _ = read_sequential_target('../train/inference/prediction/behavior_test/', True)
    predict_train = predict_train.transpose((1,0,2))
    predict_test = predict_test.transpose((1,0,2))
    predtrain_bin = predtrain_bin.transpose((1,0,2))
    predtest_bin = predtest_bin.transpose((1,0,2))
    gt = B_fw_u[:,1:,:]
    mse = np.mean(np.square(predict_test - gt))  # action loss (MSE)
    rmse = np.sqrt(mse)
    nrmse = rmse / (max_u-min_u)
    #mse = np.mean(np.square(predict_test - B_fw_u[:,1:,:]))
    print('Normalised Root-Mean squared error (NRMSE) for predicted joint values: ',nrmse*100)

if __name__ == '__main__':
    evaluate()

