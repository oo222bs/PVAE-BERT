import numpy as np
import torch
from torch import nn
from PIL import Image
from config import TrainConfig
import os, time

# Read image and turn it into 120x160
def read_input_folder(images_path,greyscale=False):
    all_file_list = []
    dir_list = []
    for (root, dirs, files) in os.walk(images_path):
        dir_list.append(root)
        file_list = []
        for file in files:
            file_list.append(os.path.join(root, file))
        file_list.sort()
        if (len(file_list) > 0):
            all_file_list.append(file_list)
    if greyscale:
        redfactor = 0.2989
        greenfactor = 0.5870
        bluefactor = 0.1140
    all_file_list.sort()
    max_len = 1
    count = 0
    all_resized_images = np.zeros((len(all_file_list), max_len, 120, 160, 3))
    for file_list in all_file_list:
        num_of_images = len(file_list)
        file_list.sort()
        resized_images = []
        for i, image in enumerate(file_list):
            original_image = Image.open(image)
            # resized_image = original_image.resize((160, 120))
            resized_image_arr = np.asarray(original_image, float) / 255.0
            if greyscale:
                resized_image_arr = resized_image_arr[:, :, 0] * redfactor + resized_image_arr[:, :, 1] * greenfactor + resized_image_arr[:, :, 2] * bluefactor
                resized_image_arr = np.expand_dims(resized_image_arr, axis=2)
            resized_images.append(resized_image_arr)
        if num_of_images > max_len:
            max_len = num_of_images
            add_zeros = np.zeros((all_resized_images.shape[0], max_len - all_resized_images.shape[1], 120, 160, 3))
            all_resized_images = np.concatenate((all_resized_images, add_zeros), axis=1)
        all_resized_images[count][:len(resized_images)] = np.asarray(resized_images)
        count += 1
    return [all_resized_images, all_file_list]


# Convolution Encoder
class ConvolutionalEncoder(nn.Module):
    def __init__(self):
        super(ConvolutionalEncoder, self).__init__()
        self.first_conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.second_conv = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.third_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.fourth_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(8, 8), stride=(5, 5), padding=2)
        self.relu = nn.ReLU()

    def forward(self, input_images):
        first_conv = self.relu(self.first_conv(input_images.float()))
        second_conv = self.relu(self.second_conv(first_conv))
        third_conv = self.relu(self.third_conv(second_conv))
        fourth_conv = self.relu(self.fourth_conv(third_conv))

        return fourth_conv


# Fully connected layers (Bottleneck)
class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        self.first_fc = nn.Linear(in_features=3*4*64, out_features=384)
        self.second_fc = nn.Linear(in_features=384, out_features=192)
        self.third_fc = nn.Linear(in_features=192, out_features=10)
        self.fourth_fc = nn.Linear(in_features=10, out_features=192)
        self.fifth_fc = nn.Linear(in_features=192, out_features=384)

    def forward(self, encoded_features):
        flattened = torch.flatten(encoded_features, start_dim=1)
        first_dense = self.first_fc(flattened)
        second_dense = self.second_fc(first_dense)
        third_dense = self.third_fc(second_dense)
        fourth_dense = self.fourth_fc(third_dense)
        fifth_dense = self.fifth_fc(fourth_dense)

        return third_dense, fifth_dense


# Deconvolutional Decoder
class DeconvolutionalDecoder(nn.Module):
    def __init__(self):
        super(DeconvolutionalDecoder, self).__init__()
        self.first_deconv = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(8, 8), stride=(5, 5), padding=2, output_padding=1)
        self.second_deconv = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.third_deconv = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.fourth_deconv = nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, dense_features):
        reshaped = torch.reshape(dense_features, (dense_features.size()[0], -1, 3, 4))
        first_deconv = self.relu(self.first_deconv(reshaped))
        second_deconv = self.relu(self.second_deconv(first_deconv))
        third_deconv = self.relu(self.third_deconv(second_deconv))
        output = self.sigmoid(self.fourth_deconv(third_deconv))
        return output


# Convolutional Autoencoder (CAE)
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        self.convolutional_encoder = ConvolutionalEncoder()
        self.bottleneck = Bottleneck()
        self.deconvolutional_decoder = DeconvolutionalDecoder()

    def forward(self, input_images):
        encoded_features = self.convolutional_encoder(input_images)
        _, dense_out = self.bottleneck(encoded_features)
        output_images = self.deconvolutional_decoder(dense_out)
        return output_images

    def extract_visual_features(self, input_image):
        encoded_features = self.convolutional_encoder(input_image)
        visual_features, _ = self.bottleneck(encoded_features)
        return visual_features

def loss(output, target):
    bce_loss = nn.BCELoss()
    return bce_loss(output.float(), target.float())

# Extract 10 dimensional visual features from images
def extract_visual_features():
    # get the training configuration
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    model = CAE().to(device)
    model.load_state_dict(torch.load(train_conf.cae_save_dir + '/cae_model.pt'))
    model.eval()
    dates = ["201207", "201223", "210107", "210115", "210129", "210203"]
    test = [2, 6, 9, 14, 17, 25, 31, 33, 34, 38, 42, 45, 50, 53, 61, 67, 69, 70,
            74, 76, 77, 78, 79, 80, 81, 84, 94, 110, 112, 113, 114, 115, 116, 117, 120, 130]
    for date in dates:
        im_data_dir = "../target_obj/image_train/" + date
        resized_input, filenames = read_input_folder(im_data_dir, False)
        batchsize = 1
        resized_input = resized_input.reshape(resized_input.shape[0], resized_input.shape[1], resized_input.shape[4], resized_input.shape[2], resized_input.shape[3])

        # Feed the dataset as input
        for i in range(resized_input.shape[0]):
            resized_input_x = resized_input[i, :, :, :, :]
            resized_input_x = torch.from_numpy(resized_input_x).to(device)
            with torch.no_grad():
                visual_features = model.extract_visual_features(resized_input_x)
            date = filenames[i][0].split(os.path.sep)[-3]
            filename = filenames[i][0].split(os.path.sep)[-2]
            if i in test:
                name = "../train/visual_feature_extraction_ori_first_test/" + date + "/" + filename + ".txt"
            else:
                name = "../train/visual_feature_extraction_ori_first/" + date + "/" + filename + ".txt"
            dir_hierarchy = name.split("/")
            dir_hierarchy = filter(lambda z: z != "..", dir_hierarchy)
            save_name = os.path.join(*dir_hierarchy)
            save_name = os.path.join("..", save_name)
            if not os.path.exists(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))
            np.savetxt(save_name, visual_features[:len(filenames[i])].cpu().numpy(), fmt="%.6f")

# Reconstruct images
def reconstruct():
    # get the training configuration
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")

    im_data_dir = train_conf.IM_dir_test
    resized_input, filenames = read_input_folder(im_data_dir)
    batchsize = 1
    resized_input = resized_input.reshape(batchsize, resized_input.shape[4], resized_input.shape[2], resized_input.shape[3])

    # Use GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    model = CAE().to(device)

    model.load_state_dict(torch.load(train_conf.cae_save_dir + '/cae_model.pt'))
    model.eval()

    # Feed the dataset as input
    for i in range(resized_input.shape[1]):
        resized_input_x = resized_input[:, i, :, :, :]
        with torch.no_grad():
            result = model.forward(resized_input_x)
        image_name = filenames[0][i].split(os.path.sep)[-1]
        name = "../train/20201123_Embodied_Language_Learning-Nico2Blocks_test/reconstructed/" + image_name
        dir_hierarchy = name.split("/")
        dir_hierarchy = filter(lambda z: z != "..", dir_hierarchy)
        save_name = os.path.join(*dir_hierarchy)
        save_name = os.path.join("..", save_name)
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        reconstructed_im = Image.fromarray(np.uint8(result[0] * 255))
        reconstructed_im.save(save_name)

# Train the CAE on the egocentric images
def train():
    # get the training configuration
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    seed = train_conf.seed
    batch_size = 200
    num_of_iterations = 5000
    # set the save directory for the model
    save_dir = train_conf.cae_save_dir
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))

    # get the dataset folder
    im_data_dir = train_conf.IM_dir
    resized_input, _ = read_input_folder(im_data_dir, False)
    resized_input_batch = resized_input.reshape(resized_input.shape[0] * resized_input.shape[1],
                                                resized_input.shape[4], resized_input.shape[2], resized_input.shape[3])
    # get the dataset folder for testing
    if train_conf.test:
        im_data_dir_test = train_conf.IM_dir_test  # get the folder for test language descriptions
        resized_input_test, _ = read_input_folder(im_data_dir_test, False)
        resized_input_test_batch = resized_input_test.reshape(resized_input_test.shape[0] * resized_input_test.shape[1],
                                                              resized_input_test.shape[4], resized_input_test.shape[2],
                                                              resized_input_test.shape[3])
    # Random Initialisation
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    model = CAE().to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimiser
    epoch_loss = []  # save a running loss
    model.train()   # tell the model that it's training time

    # Training
    previous = time.time()  # time the training
    for step in range(num_of_iterations):
        batch_idx = np.random.permutation(resized_input_batch.shape[0])[:batch_size]
        feed_resized_input_batch = resized_input_batch[batch_idx, :, :, :]
        feed_resized_input_batch = torch.from_numpy(feed_resized_input_batch).to(device)
        optimiser.zero_grad()  # free the optimizer from previous gradients
        output = model(feed_resized_input_batch)
        batch_loss = loss(output, feed_resized_input_batch)
        batch_loss.backward()  # compute gradients
        optimiser.step()  # update weights
        print("step:{} total:{}".format(step, batch_loss))

        if train_conf.test and (step + 1) % train_conf.test_interval == 0:
            batch_idx = np.random.permutation(resized_input_test_batch.shape[0])[:batch_size]
            feed_resized_input_batch = resized_input_test_batch[batch_idx, :, :, :]
            feed_resized_input_batch = torch.from_numpy(feed_resized_input_batch).to(device)
            with torch.no_grad():
                output = model(feed_resized_input_batch)
                batch_loss = loss(output, feed_resized_input_batch)
            print("test")
            print("step:{} total:{}".format(step, batch_loss))

        if (step + 1) % train_conf.log_interval == 0:
            torch.save(model.state_dict(), save_dir + '/cae_model.pt')
    past = time.time()
    print(past - previous)  # print the elapsed time


if __name__ == "__main__":
    train()
    #reconstruct()
    #extract_visual_features()
