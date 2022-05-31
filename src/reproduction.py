import torch

from pvae import PVAEBERT
from config import VaeConfig, TrainConfig
from data_util import save_latent
import numpy as np
from dataset import PairedNico2BlocksDataset
from torch.utils.data import DataLoader
from data_util import normalise, pad_with_zeros

# Reproduce the actions without descriptions
def main():
    # get the network configuration (parameters such as number of layers and units)
    paramaters = VaeConfig()
    paramaters.set_conf("../train/vae_conf.txt")

    # get the training configuration (batch size, initialisation, number of iterations, saving and loading directory)
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    save_dir = train_conf.save_dir

    # Load the dataset
    training_dataset = PairedNico2BlocksDataset(train_conf)
    test_dataset = PairedNico2BlocksDataset(train_conf, True)

    # Get the max and min values for normalisation
    max_joint = np.concatenate((training_dataset.B_bw, test_dataset.B_bw), 1).max()
    min_joint = np.concatenate((training_dataset.B_bw, test_dataset.B_bw), 1).min()
    max_vis = np.concatenate((training_dataset.V_bw, test_dataset.V_bw), 1).max()
    min_vis = np.concatenate((training_dataset.V_bw, test_dataset.V_bw), 1).min()

    # normalise the joint angles, visual features between -1 and 1 and pad the fast actions with zeros
    training_dataset.B_bw = pad_with_zeros(normalise(training_dataset.B_bw, max_joint, min_joint))
    training_dataset.B_fw = pad_with_zeros(normalise(training_dataset.B_fw, max_joint, min_joint))
    test_dataset.B_bw = pad_with_zeros(normalise(test_dataset.B_bw, max_joint, min_joint), True)
    test_dataset.B_fw = pad_with_zeros(normalise(test_dataset.B_fw, max_joint, min_joint), True)
    training_dataset.V_bw = pad_with_zeros(normalise(training_dataset.V_bw, max_vis, min_vis))
    training_dataset.V_fw = pad_with_zeros(normalise(training_dataset.V_fw, max_vis, min_vis))
    test_dataset.V_bw = pad_with_zeros(normalise(test_dataset.V_bw, max_vis, min_vis), True)
    test_dataset.V_fw = pad_with_zeros(normalise(test_dataset.V_fw, max_vis, min_vis), True)

    train_dataloader = DataLoader(training_dataset)
    test_dataloader = DataLoader(test_dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    model = PVAEBERT(paramaters).to(device)

    # Load the trained model
    checkpoint = torch.load(save_dir + '/pvae_model_bert_train.tar')       # get the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])       # load the model state

    model.eval()

    # Feed the dataset as input
    for input in train_dataloader:
        input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
        input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
        input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
        input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
        input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
        input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
        with torch.no_grad():
            result = model.reproduce_actions(input).cpu()
        result = (((result + 1) / 2) * (input["max_joint"] - input["min_joint"])) + input["min_joint"]
        save_latent(result.unsqueeze(0), input["B_filenames"][0], "reproduction")

    # Do the same for the test set
    if train_conf.test:
        for input in test_dataloader:
            input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
            input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
            input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
            input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
            input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
            input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
            with torch.no_grad():
                result = model.reproduce_actions(input).cpu()
            result = (((result + 1) / 2) * (input["max_joint"] - input["min_joint"])) + input["min_joint"]
            save_latent(result.unsqueeze(0), input["B_filenames"][0], "reproduction")

if __name__ == "__main__":
    main()
