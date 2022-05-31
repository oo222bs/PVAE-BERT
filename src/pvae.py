import torch
from torch import nn
import math
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn.functional as F

# Peephole LSTM used as encoders and decoders
class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, peephole=False, forget_bias=0.0):
        super().__init__()
        self.input_sz = input_size
        self.hidden_size = hidden_size
        self.peephole = peephole
        self.W = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size * 4))
        self.peep_i = nn.Parameter(torch.Tensor(hidden_size))
        self.peep_f = nn.Parameter(torch.Tensor(hidden_size))
        self.peep_o = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.forget_bias = forget_bias
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, sequence_len=None,
                init_states=None):
        """Assumes x is of shape (sequence, batch, feature)"""
        if sequence_len is None:
            seq_sz, bs, _ = x.size()
        else:
            seq_sz = sequence_len.max()
            _, bs, _ = x.size()
        hidden_seq = []
        if init_states is None:
            c_t, h_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            c_t, h_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[t, :, :]
            if sequence_len is not None:
                if sequence_len.min() <= t+1:
                    old_c_t = c_t.clone().detach()
                    old_h_t = h_t.clone().detach()
            # batch the computations into a single matrix multiplication
            lstm_mat = torch.cat([x_t, h_t], dim=1)
            if self.peephole:
                gates = lstm_mat @ self.W + self.bias
            else:
                gates = lstm_mat @ self.W + self.bias
                g_t = torch.tanh(gates[:, HS * 2:HS * 3])

            if self.peephole:
                i_t, j_t, f_t, o_t = (
                    (gates[:, :HS]),  # input
                    (gates[:, HS:HS * 2]),  # new input
                    (gates[:, HS * 2:HS * 3]),   # forget
                    (gates[:, HS * 3:])   # output
                )
            else:
                i_t, f_t, o_t = (
                    torch.sigmoid(gates[:, :HS]),  # input
                    torch.sigmoid(gates[:, HS:HS * 2]),# + self.forget_bias),  # forget
                    torch.sigmoid(gates[:, HS * 3:])  # output
                )

            if self.peephole:
                c_t = torch.sigmoid(f_t + self.forget_bias + c_t * self.peep_f) * c_t \
                      + torch.sigmoid(i_t + c_t * self.peep_i) * torch.tanh(j_t)
                h_t = torch.sigmoid(o_t + c_t * self.peep_o) * torch.tanh(c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)

            out = h_t.clone()
            if sequence_len is not None:
                if sequence_len.min() <= t:
                    c_t = torch.where(torch.tensor(sequence_len).to(c_t.device) <= t, old_c_t.T, c_t.T).T
                    h_t = torch.where(torch.tensor(sequence_len).to(h_t.device) <= t, old_h_t.T, h_t.T).T
                    out = torch.where(torch.tensor(sequence_len).to(out.device) <= t, torch.zeros(out.shape).to(out.device).T, out.T).T

            hidden_seq.append(out.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)

        return hidden_seq, (c_t, h_t)

# Binding loss (description, action)
def aligned_discriminative_loss(lang, act, margin=1.0):
    batch_size = int(lang.shape[0])  # number of actions
    lang_tile = torch.tile(lang, (batch_size, 1))  # replicate the descriptions according to no. of actions
    act_tile = torch.tile(act, (1, batch_size)).view(batch_size ** 2, -1)  # do the same for actions
    # Calculate the euclidean distance between paired descriptions and actions
    pair_loss = torch.sqrt(torch.sum(torch.square(lang - act), axis=1))
    all_pairs = torch.square(lang_tile - act_tile)
    loss_array = torch.sqrt(torch.sum(all_pairs, axis=1)).view(batch_size, batch_size)
    # Make representation of an action be far from that of its unpaired description
    lang_diff = torch.unsqueeze(pair_loss, axis=0) - loss_array + margin
    act_diff = torch.unsqueeze(pair_loss, axis=1) - loss_array + margin
    lang_diff = torch.maximum(lang_diff.to('cuda'), torch.zeros(lang_diff.size()).to('cuda'))
    act_diff = torch.maximum(act_diff.to('cuda'), torch.zeros(act_diff.size()).to('cuda'))
    mask = 1.0 - torch.eye(batch_size)
    lang_diff = lang_diff * mask.to('cuda')
    act_diff = act_diff * mask.to('cuda')

    return torch.mean(lang_diff) + torch.mean(act_diff) + torch.mean(pair_loss)  # return the binding loss

# Method for Losses
def loss(output, gt_description, gt_action, B_bin, net_conf):
    [L_output, B_output, L_z_mean, VB_z_mean, L_z_log_var, VB_z_log_var] = output
    # Calculate the loss
    B_output = B_output * B_bin[1:]
    L_loss = torch.mean(-torch.sum(gt_description * torch.log(L_output), 2))  # description loss
    B_loss = torch.mean(torch.square(B_output - gt_action))  # action loss (MSE)
    share_loss = aligned_discriminative_loss(L_z_mean, VB_z_mean, net_conf.delta)  # binding loss
    # add the regularisation loss (KLD)
    L_reg = L_z_log_var.exp() - L_z_log_var - 1 + L_z_mean.square()
    L_reg = torch.mean(0.5 * torch.sum(L_reg, axis=-1))
    VB_reg = VB_z_log_var.exp() - VB_z_log_var - 1 + VB_z_mean.square()
    VB_reg = torch.mean(0.5 * torch.sum(VB_reg, axis=-1))

    loss = net_conf.L_weight * L_loss + net_conf.B_weight * B_loss \
           + net_conf.S_weight * share_loss + net_conf.KL_weight*L_reg \
           + net_conf.KL_weight*VB_reg          # total loss


    return L_loss, B_loss, share_loss, loss, L_reg, VB_reg

# Model training
def train(model, batch, optimiser, epoch_loss, params):
    optimiser.zero_grad()  # free the optimizer from previous gradients
    gt_description = batch['L_fw'][1:]
    gt_action = batch['B_fw'][1:]
    output = model(batch)
    L_loss, B_loss, share_loss, batch_loss, L_reg, VB_reg = loss(output, gt_description, gt_action, batch["B_bin"], params)  # compute loss
    batch_loss.backward()  # compute gradients
    optimiser.step()  # update weights
    epoch_loss.append(batch_loss.item())  # record the batch loss

    return L_loss, B_loss, share_loss, batch_loss, L_reg, VB_reg  # return the losses

# Model validation
def validate(model, batch, epoch_loss, params):
    with torch.no_grad():
        gt_description = batch['L_fw'][1:]
        gt_action = batch['B_fw'][1:]
        output = model(batch)
        L_loss, B_loss, share_loss, batch_loss, L_reg, VB_reg = loss(output, gt_description, gt_action, batch["B_bin"], params)  # compute loss
        epoch_loss.append(batch_loss.item())  # record the batch loss

    return L_loss, B_loss, share_loss, batch_loss, L_reg, VB_reg  # return the losses

# BERT language encoder
class BertEncoder(nn.Module):
    def __init__(self, params):
        super(BertEncoder, self).__init__()
        self.params = params
        self.tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder = BertModel.from_pretrained("bert-base-uncased")      # batch, seq, hidden
        # Uncomment below if no finetuning desired
        #for param in self.encoder.parameters():
            #param.requires_grad = False

    def forward(self, inp):
        # Use the vocabulary to feed descriptions
        file = open('../vocabulary.txt', 'r')
        vocab = file.read().splitlines()
        file.close()
        t = inp[:, :, :].argmax(axis=-1)
        descriptions = []
        for i in range(inp.shape[1]):
            sentence = ''
            for k in range(1, inp.shape[0] - 1):
                sentence += vocab[t[k, i]] + ' '
            descriptions.append(sentence[:-1])
            # Different versions of the description:
            #descriptions.append(sentence.split(' ')[-2] + ' ' + sentence.split(' ')[0] + ' ' + sentence.split(' ')[1])
            #descriptions.append('could you ' + sentence[:-1] + '?')
            #descriptions.append(sentence.split(' ')[0] + ' ' + sentence.split(' ')[-2] + ' ' + sentence.split(' ')[1] + ' ' + sentence.split(' ')[2] + ' ' + sentence.split(' ')[3] + ' ' + sentence.split(' ')[4])
            #descriptions.append(sentence.split(' ')[-2] + ' ' + sentence.split(' ')[1] + ' ' + sentence.split(' ')[2] + ' ' + sentence.split(' ')[3] + ' ' + sentence.split(' ')[4] + ' ' + sentence.split(' ')[0])
            #descriptions.append(sentence.split(' ')[1] + ' ' + sentence.split(' ')[2] + ' ' + sentence.split(' ')[3] + ' ' + sentence.split(' ')[4] +' ' + sentence.split(' ')[5] + ' ' + sentence.split(' ')[0])
        encoded_input = self.tokeniser(descriptions, return_tensors='pt', padding=True)
        output = self.encoder(**encoded_input.to(self.encoder.device))
        v = self.mean_pooling(output, encoded_input['attention_mask'])     # use this when descriptions have different number of subwords to ignore padding
        #v = output.last_hidden_state.mean(1)
        return v

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Language and Action Encoders
class Encoder(nn.Module):
    def __init__(self, params, lang=False):
        super(Encoder, self).__init__()
        self.params = params
        if lang:
            self.enc_cells = torch.nn.Sequential()
            for i in range(self.params.L_num_layers):
                if i == 0:
                    self.enc_cells.add_module("ellstm"+str(i), PeepholeLSTM(input_size=self.params.L_input_dim,
                                                                            hidden_size=self.params.L_num_units,
                                                                            peephole=True, forget_bias=0.8))
                else:
                    self.enc_cells.add_module("ellstm"+str(i), PeepholeLSTM(input_size=self.params.L_num_units,
                                                                            hidden_size=self.params.L_num_units,
                                                                            peephole=True, forget_bias=0.8))
        else:
            self.enc_cells = torch.nn.Sequential()
            for i in range(self.params.VB_num_layers):
                if i == 0:
                    self.enc_cells.add_module("ealstm" + str(i), PeepholeLSTM(input_size=self.params.VB_input_dim,
                                                                              hidden_size=self.params.VB_num_units,
                                                                              peephole=True, forget_bias=0.8))
                else:
                    self.enc_cells.add_module("ealstm" + str(i), PeepholeLSTM(input_size=self.params.VB_num_units,
                                                                              hidden_size=self.params.VB_num_units,
                                                                              peephole=True, forget_bias=0.8))

    def forward(self, inp, sequence_length, lang=False):
        if lang:
            num_of_layers = self.params.L_num_layers
        else:
            num_of_layers = self.params.VB_num_layers
        layer_input = inp
        states = []
        for l in range(num_of_layers):
            enc_cell = self.enc_cells.__getitem__(l)
            hidden_seq, (cn, hn) = enc_cell(layer_input.float().to('cuda'), sequence_len=sequence_length)
            layer_input = hidden_seq
            states.append((cn, hn))
        states = tuple(map(torch.stack, zip(*states)))
        final_state = torch.stack(states, dim=1)    # n_layers, 2, batch_size, n_units
        final_state = final_state.permute(2,0,1,3)  # transpose to batchsize, n_layers, 2, n_units
        final_state = torch.reshape(final_state, (int(final_state.shape[0]), -1))
        return final_state

# Language and Action Decoders
class Decoder(nn.Module):
    def __init__(self, params, lang=False):
        super(Decoder, self).__init__()
        self.params = params
        if lang:
            self.dec_cells = torch.nn.Sequential()
            for i in range(self.params.L_num_layers):
                if i == 0:
                    self.dec_cells.add_module("dllstm"+str(i), PeepholeLSTM(self.params.L_input_dim,
                                                                            self.params.L_num_units, True, forget_bias=0.8).to('cuda'))
                else:
                    self.dec_cells.add_module("dllstm"+str(i), PeepholeLSTM(self.params.L_num_units,
                                                                            self.params.L_num_units, True, forget_bias=0.8).to('cuda'))
            self.linear = nn.Linear(self.params.L_num_units, self.params.L_input_dim)
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.dec_cells = torch.nn.Sequential()
            for i in range(self.params.VB_num_layers):
                if i == 0:
                    self.dec_cells.add_module("dalstm"+str(i), PeepholeLSTM(input_size=self.params.VB_input_dim,
                                                                            hidden_size=self.params.VB_num_units,
                                                                            peephole=True, forget_bias=0.8).to('cuda'))
                else:
                    self.dec_cells.add_module("dalstm"+str(i), PeepholeLSTM(input_size=self.params.VB_num_units,
                                                                            hidden_size=self.params.VB_num_units,
                                                                            peephole=True, forget_bias=0.8).to('cuda'))
            self.linear = nn.Linear(self.params.VB_num_units, self.params.B_input_dim)
            self.tanh = nn.Tanh()

    def forward(self, input, length, initial_state=None, lang=False):
        y = []

        if lang:
            initial_state = initial_state.view(initial_state.size()[0], self.params.L_num_layers, 2, self.params.L_num_units)
            initial_state = initial_state.permute(1, 2, 0, 3)
            for i in range(length - 1):
                dec_states = []
                layer_input = input.unsqueeze(0)
                if i == 0:
                    for j in range(self.params.L_num_layers):
                        dec_cell = self.dec_cells.__getitem__(j)
                        dec_state = (initial_state[j][0].float().to('cuda'), initial_state[j][1].float().to('cuda'))
                        output, (cx, hx) = dec_cell(layer_input.float().to('cuda'), init_states=dec_state)
                        dec_state = (cx, hx)
                        dec_states.append(dec_state)
                        layer_input = output
                else:
                    layer_input = out
                    for j in range(self.params.L_num_layers):
                        dec_cell = self.dec_cells.__getitem__(j)
                        dec_state = prev_dec_states[j]
                        output, (cx, hx) = dec_cell(layer_input, init_states=dec_state)
                        dec_state = (cx, hx)
                        dec_states.append(dec_state)
                        layer_input = output
                prev_dec_states = dec_states
                linear = self.linear(layer_input)
                out = self.softmax(linear)
                y.append(out.squeeze())
        else:
            initial_state = initial_state.view(initial_state.size()[0], self.params.VB_num_layers, 2, self.params.VB_num_units)
            initial_state = initial_state.permute(1, 2, 0, 3)
            for i in range(length - 1):
                current_V_in = input[0][i]
                dec_states = []
                if i == 0:
                    #current_V_in = input[0][0]
                    current_B_in = input[-1]
                    layer_input = torch.cat([current_V_in, current_B_in], dim=1).unsqueeze(0)
                    for j in range(self.params.VB_num_layers):
                        dec_state = (initial_state[j][0].float(), initial_state[j][1].float())
                        dec_cell = self.dec_cells.__getitem__(j)
                        output, (cx, hx) = dec_cell(layer_input.float(), init_states=dec_state)
                        dec_state = (cx, hx)
                        dec_states.append(dec_state)
                        layer_input = output
                else:
                    #current_V_in = torch.zeros(current_V_in.size(0), 30).cuda()
                    current_B_in = out.squeeze(dim=0)
                    layer_input = torch.cat([current_V_in, current_B_in], dim=1).unsqueeze(0)
                    for j in range(self.params.VB_num_layers):
                        dec_cell = self.dec_cells.__getitem__(j)
                        dec_state = prev_dec_states[j]
                        output, (cx, hx) = dec_cell(layer_input.float(), init_states=dec_state)
                        dec_state = (cx, hx)
                        dec_states.append(dec_state)
                        layer_input = output
                prev_dec_states = dec_states
                linear = self.linear(layer_input)
                out = self.tanh(linear)
                y.append(out.squeeze())
        y = torch.stack(y, dim=0)
        return y

# Paired Variational Autoencoders
class PVAE(nn.Module):
    def __init__(self, params):
        super(PVAE, self).__init__()
        self.params = params      # obtain the parameters

        #Encoders
        self.lang_encoder = Encoder(self.params, True)
        self.action_encoder = Encoder(self.params, False)

        # Bottleneck
        self.lang_mean = nn.Linear(self.params.L_num_units*self.params.L_num_layers*2, self.params.S_dim)
        self.lang_log_variance = nn.Linear(self.params.L_num_units*self.params.L_num_layers*2, self.params.S_dim)
        self.action_mean = nn.Linear(self.params.VB_num_units*self.params.VB_num_layers*2, self.params.S_dim)
        self.action_log_variance = nn.Linear(self.params.VB_num_units*self.params.VB_num_layers*2, self.params.S_dim)

        # Initial states
        self.initial_lang = nn.Linear(self.params.S_dim, self.params.L_num_units*self.params.L_num_layers*2)
        self.initial_act = nn.Linear(self.params.S_dim, self.params.VB_num_units*self.params.VB_num_layers*2)

        # Decoders
        self.lang_decoder = Decoder(self.params, True)
        self.action_decoder = Decoder(self.params, False)

    def forward(self, inp):
        # Encode language
        encoded_lang = self.lang_encoder(inp['L_bw'], inp['L_len'].int().numpy(), True)

        # Encode action
        VB_input = torch.cat([inp['V_bw'], inp['B_bw']], dim=2)            # Prepare vision-joint angle input
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        VB_input_f = inp['VB_fw']           #Initial input for action decoder

        # Get means and variances
        L_z_mean = self.lang_mean(encoded_lang)
        L_z_log_var = self.lang_log_variance(encoded_lang)
        VB_z_mean = self.action_mean(encoded_act)
        VB_z_log_var = self.action_log_variance(encoded_act)

        # Get z_lang and z_act using the reparameterisation trick
        L_sampling = self.reparameterise(L_z_mean, L_z_log_var)
        VB_sampling = self.reparameterise(VB_z_mean, VB_z_log_var)

        # Get initial states for decoders
        L_dec_init_state = self.initial_lang(L_sampling)
        VB_dec_init_state = self.initial_act(VB_sampling)

        # Reproduce the inputs
        L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
        B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)

        return L_output, B_output, L_z_mean, VB_z_mean, L_z_log_var, VB_z_log_var

    def reparameterise(self, mean, log_var):
        std = torch.exp(log_var*0.5)
        epsilon = torch.normal(mean=0.0, std=0.1, size=(mean.size()[0], mean.size()[1])).to('cuda')
        return mean + std * epsilon

    def extract_representations(self, inp):
        encoded_lang = self.lang_encoder(inp['L_bw'], inp['L_len'].int().numpy(), True)
        VB_input = torch.cat([inp['V_bw'], inp['B_bw']], dim=2)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        L_z_mean = self.lang_mean(encoded_lang)
        VB_z_mean = self.action_mean(encoded_act)

        return L_z_mean, VB_z_mean

    def language_to_action(self, inp):
        encoded_lang = self.lang_encoder(inp['L_bw'], inp['L_len'].int().numpy(), True)
        VB_input_f = inp['VB_fw']

        L_z_mean = self.lang_mean(encoded_lang)
        VB_dec_init_state = self.initial_act(L_z_mean)

        B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)

        return B_output

    def action_to_language(self, inp):
        VB_input = torch.cat([inp['V_bw'], inp['B_bw']], dim=2)

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        VB_z_mean = self.action_mean(encoded_act)
        L_dec_init_state = self.initial_lang(VB_z_mean)

        L_output = self.lang_decoder(inp['L_fw'][0], int(inp['L_len'].item()), L_dec_init_state, True)

        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])

        return one_hot

    def reproduce_actions(self, inp):
        VB_input = torch.cat([inp['V_bw'], inp['B_bw']], dim=2)
        VB_input_f = inp['VB_fw']
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        VB_z_mean = self.action_mean(encoded_act)
        VB_dec_init_state = self.initial_act(VB_z_mean)

        B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)

        return B_output

    def reproduce_lang(self, inp):
        encoded_lang = self.lang_encoder(inp['L_bw'], inp['L_len'].int().numpy(), True)

        L_z_mean = self.lang_mean(encoded_lang)
        L_dec_init_state = self.initial_lang(L_z_mean)

        L_output = self.lang_decoder(inp['L_fw'][0], int(inp['L_len'].item()), L_dec_init_state, True)

        max_ind = torch.argmax(nn.functional.softmax(L_output, dim=-1), -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])

        return one_hot

# PVAE-BERT
class PVAEBERT(nn.Module):
    def __init__(self, params):
        super(PVAEBERT, self).__init__()

        self.params = params # obtain the parameters

        # Encoders
        self.lang_encoder = BertEncoder(self.params)
        self.action_encoder = Encoder(self.params, False)

        # Bottleneck
        self.lang_mean = nn.Linear(768, self.params.S_dim)
        self.lang_log_variance = nn.Linear(768, self.params.S_dim)
        self.action_mean = nn.Linear(self.params.VB_num_units*self.params.VB_num_layers*2, self.params.S_dim)
        self.action_log_variance = nn.Linear(self.params.VB_num_units*self.params.VB_num_layers*2, self.params.S_dim)

        # Initial states for decoders
        self.initial_lang = nn.Linear(self.params.S_dim, self.params.L_num_units*self.params.L_num_layers*2)
        self.initial_act = nn.Linear(self.params.S_dim, self.params.VB_num_units*self.params.VB_num_layers*2)

        # Decoders
        self.lang_decoder = Decoder(self.params, True)
        self.action_decoder = Decoder(self.params, False)

    def forward(self, inp):

        # Encode language
        encoded_lang = self.lang_encoder(inp['L_fw'])

        # Encode action
        VB_input = torch.cat([inp['V_bw'], inp['B_bw']], dim=2)  # Prepare vision-joint angle input
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        VB_input_f = inp['VB_fw']  # Initial input for action decoder

        # Get means and variances
        L_z_mean = self.lang_mean(encoded_lang)
        L_z_log_var = self.lang_log_variance(encoded_lang)
        VB_z_mean = self.action_mean(encoded_act)
        VB_z_log_var = self.action_log_variance(encoded_act)

        # Get z_lang and z_act using the reparameterisation trick
        L_sampling = self.reparameterise(L_z_mean, L_z_log_var)
        VB_sampling = self.reparameterise(VB_z_mean, VB_z_log_var)

        # Get initial states for decoders
        L_dec_init_state = self.initial_lang(L_sampling)
        VB_dec_init_state = self.initial_act(VB_sampling)

        # Reproduce the inputs
        L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
        B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)

        return L_output, B_output, L_z_mean, VB_z_mean, L_z_log_var, VB_z_log_var

    def reparameterise(self, mean, log_var):
        std = torch.exp(log_var*0.5)
        epsilon = torch.normal(mean=0.0, std=0.1, size=(mean.size()[0], mean.size()[1])).to('cuda')

        return mean + std * epsilon

    def extract_representations(self, inp):

        encoded_lang = self.lang_encoder(inp['L_fw'])
        VB_input = torch.cat([inp['V_bw'], inp['B_bw']], dim=2)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        L_z_mean = self.lang_mean(encoded_lang)
        VB_z_mean = self.action_mean(encoded_act)

        return L_z_mean, VB_z_mean

    def language_to_action(self, inp):
        encoded_lang = self.lang_encoder(inp['L_fw'])

        VB_input_f = inp['VB_fw']

        L_z_mean = self.lang_mean(encoded_lang)
        VB_dec_init_state = self.initial_act(L_z_mean)

        B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)

        return B_output

    def action_to_language(self, inp):
        VB_input = torch.cat([inp['V_bw'], inp['B_bw']], dim=2)

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        VB_z_mean = self.action_mean(encoded_act)
        L_dec_init_state = self.initial_lang(VB_z_mean)

        L_output = self.lang_decoder(inp['L_fw'][0], int(inp['L_len'].item()), L_dec_init_state, True)

        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])

        return one_hot

    def reproduce_actions(self, inp):
        VB_input = torch.cat([inp['V_bw'], inp['B_bw']], dim=2)
        VB_input_f = inp['VB_fw']
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        VB_z_mean = self.action_mean(encoded_act)
        VB_dec_init_state = self.initial_act(VB_z_mean)

        B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)

        return B_output

    def reproduce_lang(self, inp):
        encoded_lang = self.lang_encoder(inp['L_fw'])

        L_z_mean = self.lang_mean(encoded_lang)
        L_dec_init_state = self.initial_lang(L_z_mean)

        L_output = self.lang_decoder(inp['L_fw'][0], int(inp['L_len'].item()), L_dec_init_state, True)

        max_ind = torch.argmax(nn.functional.softmax(L_output, dim=-1), -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])

        return one_hot