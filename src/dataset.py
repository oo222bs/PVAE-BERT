from torch.utils.data import Dataset
from data_util import read_sequential_target

class PairedNico2BlocksDataset(Dataset):
    def __init__(self, dataset_dirs, test=False):
        # get the dataset folders
        if test:
            lang_dir = dataset_dirs.L_dir_test
            joints_dir = dataset_dirs.B_dir_test
            vis_dir = dataset_dirs.V_dir_test
        else:
            lang_dir = dataset_dirs.L_dir
            joints_dir = dataset_dirs.B_dir
            vis_dir = dataset_dirs.V_dir

        # get the descriptions
        self.L_fw, self.L_bw, self.L_bin, self.L_len, self.L_filenames = read_sequential_target(lang_dir, True)

        # get the joint angles for actions
        self.B_fw, self.B_bw, self.B_bin, self.B_len, self.B_filenames = read_sequential_target(joints_dir, True)

        # before normalisation save max and min joint angles to variables (will be used when converting norm to original values)
        self.maximum_joint = self.B_fw.max()
        self.minimum_joint = self.B_fw.min()

        # get the visual features for action images
        self.V_fw, self.V_bw, self.V_bin, self.V_len = read_sequential_target(vis_dir)

        # create variables for data shapes
        self.L_shape = (self.L_fw.shape[0] // 8, self.L_fw.shape[1], self.L_fw.shape[2])
        self.B_shape = self.B_fw.shape
        self.V_shape = self.V_fw.shape

    def __len__(self):
        return len(self.L_len)

    def __getitem__(self, index):
        items = {}
        items["L_fw"] = self.L_fw[:, index, :]
        items["L_bw"] = self.L_bw[:, index, :]
        items["B_fw"] = self.B_fw[:, index, :]
        items["B_bw"] = self.B_bw[:, index, :]
        items["V_fw"] = self.V_fw[:, index, :]
        items["V_bw"] = self.V_bw[:, index, :]
        items["L_len"] = self.L_len[index] / 8     # 8 alternatives per description
        items["B_len"] = self.B_len[index]
        items["V_len"] = self.V_len[index]
        items["B_bin"] = self.B_bin[:, index, :]
        items["L_filenames"] = self.L_filenames[index]
        items["B_filenames"] = self.B_filenames[index]
        items["max_joint"] = self.maximum_joint
        items["min_joint"] = self.minimum_joint
        return items

