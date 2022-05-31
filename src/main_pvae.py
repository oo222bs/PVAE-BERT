import torch
from config import VaeConfig, TrainConfig
import os
import numpy as np
from pvae import PVAE, PVAEBERT, train, validate
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from dataset import PairedNico2BlocksDataset
from data_util import normalise, pad_with_zeros
import datetime


def main():
    # get the network configuration (parameters such as number of layers and units)
    paramaters = VaeConfig()
    paramaters.set_conf("../train/vae_conf.txt")

    # get the training configuration
    # (batch size, initialisation, num_of_iterations number, saving and loading directory)
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    seed = train_conf.seed
    batch_size = train_conf.batch_size
    num_of_iterations = train_conf.num_of_iterations
    learning_rate = train_conf.learning_rate
    save_dir = train_conf.save_dir
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))

    # Random Initialisation
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Use GPU if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    print("The currently selected GPU is number:", torch.cuda.current_device(),
          ", it's a ", torch.cuda.get_device_name(device=None))
    # Create a model instance
    model = PVAEBERT(paramaters).to(device)
    # Initialise the optimiser
    #optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)      # Adam optimiser

    #  Inspect the model with tensorboard
    model_name = 'pvae_model_bert_train'
    date = str(datetime.datetime.now()).split('.')[0]
    writer = SummaryWriter(log_dir='.././logs/'+model_name+date)  # initialize the writer with folder "./logs"

    # Load the trained model
    #checkpoint = torch.load(save_dir + '/pvae_model_bert_train.tar')       # get the checkpoint
    #model.load_state_dict(checkpoint['model_state_dict'])       # load the model state
    #optimiser.load_state_dict(checkpoint['optimiser_state_dict'])   # load the optimiser state

    epoch_loss = []  # save a running loss
    model.train()  # tell the model that it's training time

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

    # Load the training and testing sets with DataLoader
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    # Training
    for step in range(num_of_iterations):
        input = next(iter(train_dataloader))
        input["L_fw"] = input["L_fw"].transpose(0, 1)
        input["L_bw"] = input["L_bw"].transpose(0, 1)
        input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
        input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
        input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
        input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
        input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
        input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
        sentence_idx = np.random.randint(8)  # Generate random index for description alternatives

        # Choose one of the eight description alternatives according to the generated random index
        L_fw_feed = input["L_fw"][0 + 5 * sentence_idx:5 + 5 * sentence_idx, :, :]
        L_bw_feed = input["L_bw"][35 - 5 * sentence_idx:40 - 5 * sentence_idx, :, :]

        input["L_fw"] = L_fw_feed.to(device)
        input["L_bw"] = L_bw_feed.to(device)

        # Train and print the losses
        l, b, s, t, lr, vbr = train(model, input, optimiser, epoch_loss, paramaters)
        print("step:{} total:{}, language:{}, behavior:{}, share:{}, language_reg:{},"
              " behavior_reg:{}".format(step, t, l, b, s, lr, vbr))

        writer.add_scalar('Training Loss', np.mean(epoch_loss), step)     # add the overall loss to the Tensorboard

        #scheduler.step(np.mean(epoch_loss))
        # Testing
        if train_conf.test and (step+1) % train_conf.test_interval == 0:
            input = next(iter(test_dataloader))
            input["L_fw"] = input["L_fw"].transpose(0, 1)
            input["L_bw"] = input["L_bw"].transpose(0, 1)
            input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
            input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
            input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
            input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
            input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
            input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
            sentence_idx = np.random.randint(8)  # Generate random index for description alternatives
            # Choose one of the eight description alternatives
            L_fw_feed = input["L_fw"][0 + 5 * sentence_idx:5 + 5 * sentence_idx, :, :]
            L_bw_feed = input["L_bw"][35 - 5 * sentence_idx:40 - 5 * sentence_idx, :, :]

            input["L_fw"] = L_fw_feed.to(device)
            input["L_bw"] = L_bw_feed.to(device)

            # Calculate and print the losses
            l, b, s, t, lr, vbr = validate(model, input, epoch_loss, paramaters)
            print("test")
            print("step:{} total:{}, language:{}, behavior:{}, share:{}, "
                  "language_reg:{}, behavior_reg:{}".format(step, t, l, b, s, lr, vbr))
            writer.add_scalar('Test Loss', np.mean(epoch_loss), step)  # add the overall loss to the Tensorboard

        # Save the model parameters at every log interval
        if (step+1) % train_conf.log_interval == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict()},
                       save_dir + '/pvae_model_bert_train.tar')
    # Flush and close the summary writer of Tensorboard
    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()


