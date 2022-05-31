import torch

from pvae import PVAEBERT, PVAE
from config import VaeConfig, TrainConfig
from data_util import save_latent
import numpy as np
from dataset import PairedNico2BlocksDataset
from torch.utils.data import DataLoader

# Find the descriptions via given actions
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

    train_dataloader = DataLoader(training_dataset)
    test_dataloader = DataLoader(test_dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    model = PVAE(paramaters).to(device)

    # Load the trained model
    checkpoint = torch.load(save_dir + '/pvae_model.tar')       # get the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])       # load the model state

    model.eval()
    file = open('../vocabulary.txt', 'r')
    vocab = file.read().splitlines()
    # Feed the dataset as input
    for input in train_dataloader:
        L_fw_before = input["L_fw"].transpose(0, 1)
        L_bw_before  = input["L_bw"].transpose(0, 1)
        input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
        input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
        input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
        input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
        sentence_idx = np.random.randint(8)
        L_fw_feed = L_fw_before[0 + 5 * sentence_idx:5 + 5 * sentence_idx, :, :]
        L_bw_feed = L_bw_before[35 - 5 * sentence_idx:40 - 5 * sentence_idx, :, :]
        input["L_fw"] = L_fw_feed.to(device)
        input["L_bw"] = L_bw_feed.to(device)

        L_fw_before = L_fw_before.numpy()
        with torch.no_grad():
            result = model.reproduce_lang(input).cpu()
        save_latent(result.unsqueeze(0), input["L_filenames"][0], "lang2lang")  # save the predicted descriptions
        r = result.argmax(axis=1).numpy()
        t = L_fw_before[1:5, 0, :].argmax(axis=1)
        t_second = L_fw_before[6:10, 0, :].argmax(axis=1)
        t_third = L_fw_before[11:15, 0, :].argmax(axis=1)
        t_fourth = L_fw_before[16:20, 0, :].argmax(axis=1)
        t_fifth = L_fw_before[21:25, 0, :].argmax(axis=1)
        t_sixth = L_fw_before[26:30, 0, :].argmax(axis=1)
        t_seventh = L_fw_before[31:35, 0, :].argmax(axis=1)
        t_eighth = L_fw_before[36:40, 0, :].argmax(axis=1)

        # Check if predicted descriptions match the original ones
        if (r == t).all() or (r == t_second).all() or (r == t_third).all() or (r == t_fourth).all() \
                or (r == t_fifth).all() or (r == t_sixth).all() or (r == t_seventh).all() or (r == t_eighth).all():
            print(True)  # Check if predicted descriptions match the original ones
        else:
            print(False)
            print("Expected:", end=" ")
            for k in range(r.size):
                print(vocab[t[k]], end=" ")
            print()
        for k in range(r.size):
            print(vocab[r[k]], end=" ")
        print()
    # Do the same for the test set
    if train_conf.test:
        print("test!")
        for input in test_dataloader:
            L_fw_before = input["L_fw"].transpose(0, 1)
            L_bw_before = input["L_bw"].transpose(0, 1)
            input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
            input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
            input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
            input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
            sentence_idx = np.random.randint(8)
            L_fw_feed = L_fw_before[0 + 5 * sentence_idx:5 + 5 * sentence_idx, :, :]
            L_bw_feed = L_bw_before[35 - 5 * sentence_idx:40 - 5 * sentence_idx, :, :]
            input["L_fw"] = L_fw_feed.to(device)
            input["L_bw"] = L_bw_feed.to(device)

            L_fw_before = L_fw_before.numpy()
            with torch.no_grad():
                result = model.reproduce_lang(input).cpu()
            save_latent(result.unsqueeze(0), input["L_filenames"][0],
                        "lang2lang")  # save the predicted descriptions
            r = result.argmax(axis=1).numpy()
            t = L_fw_before[1:5, 0, :].argmax(axis=1)
            t_second = L_fw_before[6:10, 0, :].argmax(axis=1)
            t_third = L_fw_before[11:15, 0, :].argmax(axis=1)
            t_fourth = L_fw_before[16:20, 0, :].argmax(axis=1)
            t_fifth = L_fw_before[21:25, 0, :].argmax(axis=1)
            t_sixth = L_fw_before[26:30, 0, :].argmax(axis=1)
            t_seventh = L_fw_before[31:35, 0, :].argmax(axis=1)
            t_eighth = L_fw_before[36:40, 0, :].argmax(axis=1)

            # Check if predicted descriptions match the original ones
            if (r == t).all() or (r == t_second).all() or (r == t_third).all() or (r == t_fourth).all() \
                    or (r == t_fifth).all() or (r == t_sixth).all() or (r == t_seventh).all() or (r == t_eighth).all():
                print(True)  # Check if predicted descriptions match the original ones
            else:
                print(False)
                print("Expected:", end=" ")
                for k in range(r.size):
                    print(vocab[t[k]], end=" ")
                print()
            for k in range(r.size):
                print(vocab[r[k]], end=" ")
            print()

if __name__ == "__main__":
    main()