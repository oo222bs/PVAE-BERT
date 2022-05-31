import torch

from pvae import PVAEBERT, PVAE
from config import VaeConfig, TrainConfig
from data_util import save_latent
import numpy as np
from dataset import PairedNico2BlocksDataset
from torch.utils.data import DataLoader
from data_util import normalise, pad_with_zeros

# Extract shared representations, viz. binding layer features
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
    test_dataset.B_bw = pad_with_zeros(normalise(test_dataset.B_bw, max_joint, min_joint), True)
    training_dataset.V_bw = pad_with_zeros(normalise(training_dataset.V_bw, max_vis, min_vis))
    test_dataset.V_bw = pad_with_zeros(normalise(test_dataset.V_bw, max_vis, min_vis), True)

    train_dataloader = DataLoader(training_dataset)
    test_dataloader = DataLoader(test_dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    model = PVAEBERT(paramaters).to(device)

    # Load the trained model
    checkpoint = torch.load(save_dir + '/pvae_model_bert_train.tar')  # get the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # load the model state

    model.eval()

    # Feed the dataset as input
    for input in train_dataloader:
        sentence_idx = np.random.randint(8)              # Random index for description alternatives
        input["L_fw"] = input["L_fw"].transpose(0, 1)
        input["L_bw"] = input["L_bw"].transpose(0,1)
        input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
        input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
        # Choose one of eight description alternatives
        if sentence_idx == 0:
            L_fw_feed = input["L_fw"][0:5, :, :]
            L_bw_feed = input["L_bw"][35:40, :, :]
        elif sentence_idx == 1:
            L_fw_feed = input["L_fw"][5:10, :, :]
            L_bw_feed = input["L_bw"][30:35, :, :]
        elif sentence_idx == 2:
            L_fw_feed = input["L_fw"][10:15, :, :]
            L_bw_feed = input["L_bw"][25:30, :, :]
        elif sentence_idx == 3:
            L_fw_feed = input["L_fw"][15:20, :, :]
            L_bw_feed = input["L_bw"][20:25, :, :]
        elif sentence_idx == 4:
            L_fw_feed = input["L_fw"][20:25, :, :]
            L_bw_feed = input["L_bw"][15:20, :, :]
        elif sentence_idx == 5:
            L_fw_feed = input["L_fw"][25:30, :, :]
            L_bw_feed = input["L_bw"][10:15, :, :]
        elif sentence_idx == 6:
            L_fw_feed = input["L_fw"][30:35, :, :]
            L_bw_feed = input["L_bw"][5:10, :, :]
        else:
            L_fw_feed = input["L_fw"][35:40, :, :]
            L_bw_feed = input["L_bw"][0:5, :, :]

        input["L_fw"] = L_fw_feed.to(device)
        input["L_bw"] = L_bw_feed.to(device)

        with torch.no_grad():
            L_enc, VB_enc = model.extract_representations(input)
        save_latent(L_enc.cpu(), input["L_filenames"][0], dirname='latent')             # save the binding features for description
        save_latent(VB_enc.cpu(), input["B_filenames"][0], dirname='latent')            # save the binding features for actions

    # Do the same for the test set
    if train_conf.test:
        for input in test_dataloader:
            input["L_fw"] = input["L_fw"].transpose(0, 1)
            input["L_bw"] = input["L_bw"].transpose(0, 1)
            input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
            input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
            sentence_idx = 0#np.random.randint(8)
            # Choose one of eight description alternatives
            if sentence_idx == 0:
                L_fw_feed = input["L_fw"][0:5, :, :]
                L_bw_feed = input["L_bw"][35:40, :, :]
            elif sentence_idx == 1:
                L_fw_feed = input["L_fw"][5:10, :, :]
                L_bw_feed = input["L_bw"][30:35, :, :]
            elif sentence_idx == 2:
                L_fw_feed = input["L_fw"][10:15, :, :]
                L_bw_feed = input["L_bw"][25:30, :, :]
            elif sentence_idx == 3:
                L_fw_feed = input["L_fw"][15:20, :, :]
                L_bw_feed = input["L_bw"][20:25, :, :]
            elif sentence_idx == 4:
                L_fw_feed = input["L_fw"][20:25, :, :]
                L_bw_feed = input["L_bw"][15:20, :, :]
            elif sentence_idx == 5:
                L_fw_feed = input["L_fw"][25:30, :, :]
                L_bw_feed = input["L_bw"][10:15, :, :]
            elif sentence_idx == 6:
                L_fw_feed = input["L_fw"][30:35, :, :]
                L_bw_feed = input["L_bw"][5:10, :, :]
            else:
                L_fw_feed = input["L_fw"][35:40, :, :]
                L_bw_feed = input["L_bw"][0:5, :, :]

            input["L_fw"] = L_fw_feed.to(device)
            input["L_bw"] = L_bw_feed.to(device)
            with torch.no_grad():
                L_enc, VB_enc = model.extract_representations(input)
            save_latent(L_enc.cpu(), input["L_filenames"][0], dirname='latent')        # save the binding features for description
            save_latent(VB_enc.cpu(), input["B_filenames"][0], dirname='latent')       # save the binding features for actions

if __name__ == "__main__":
    main()
