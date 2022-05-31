import torch
from pvae import PVAEBERT, PVAE
from config import VaeConfig, TrainConfig
from data_util import save_latent
import numpy as np
from dataset import PairedNico2BlocksDataset
from torch.utils.data import DataLoader
from data_util import normalise, pad_with_zeros

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
    training_data = PairedNico2BlocksDataset(train_conf)
    test_data = PairedNico2BlocksDataset(train_conf, True)

    # Get the max and min values for normalisation
    max_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).max()
    min_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).min()
    max_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).max()
    min_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).min()

    # normalise the joint angles, visual features between -1 and 1 and pad the fast actions with zeros
    training_data.B_bw = pad_with_zeros(normalise(training_data.B_bw, max_joint, min_joint))
    training_data.B_fw = pad_with_zeros(normalise(training_data.B_fw, max_joint, min_joint))
    test_data.B_bw = pad_with_zeros(normalise(test_data.B_bw, max_joint, min_joint), True)
    test_data.B_fw = pad_with_zeros(normalise(test_data.B_fw, max_joint, min_joint), True)
    training_data.V_bw = pad_with_zeros(normalise(training_data.V_bw, max_vis, min_vis))
    training_data.V_fw = pad_with_zeros(normalise(training_data.V_fw, max_vis, min_vis))
    test_data.V_bw = pad_with_zeros(normalise(test_data.V_bw, max_vis, min_vis), True)
    test_data.V_fw = pad_with_zeros(normalise(test_data.V_fw, max_vis, min_vis), True)

    train_dataloader = DataLoader(training_data)
    test_dataloader = DataLoader(test_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    model = PVAEBERT(paramaters).to(device)

    # Load the trained model
    checkpoint = torch.load(save_dir + '/pvae_model_bert_train.tar')       # get the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])       # load the model state

    model.eval()
    file = open('../vocabulary.txt', 'r')
    vocab = file.read().splitlines()
    # Feed the dataset as input
    for input in train_dataloader:
        L_fw_before = input["L_fw"].transpose(0, 1)
        sentence_idx = np.random.randint(8)
        input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
        input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
        input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
        input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)

        L_fw_feed = L_fw_before[0 + 5 * sentence_idx:5 + 5 * sentence_idx, :, :]
        input["L_fw"] = L_fw_feed.to(device)
        L_fw_before = L_fw_before.numpy()
        with torch.no_grad():
            result = model.action_to_language(input).cpu()
        save_latent(result.unsqueeze(0), input["L_filenames"][0], "recognition")  # save the predicted descriptions
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
            print(True)
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
            sentence_idx = np.random.randint(8)
            input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
            input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
            input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
            input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)

            L_fw_feed = L_fw_before[0 + 5 * sentence_idx:5 + 5 * sentence_idx, :, :]
            input["L_fw"] = L_fw_feed.to(device)
            L_fw_before = L_fw_before.numpy()
            with torch.no_grad():
                result = model.action_to_language(input).cpu()
            save_latent(result.unsqueeze(0), input["L_filenames"][0],
                        "recognition")  # save the predicted descriptions
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
                print(True)
            else:
                print(False)
                print("Expected: ", end=" ")
                for k in range(r.size):
                    print(vocab[t[k]], end=" ")
                print()
            for k in range(r.size):
                print(vocab[r[k]], end=" ")
            print()

if __name__ == "__main__":
    main()
