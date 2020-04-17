import collections
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app
from torch.utils.data import DataLoader, Dataset
from datasets import IMDBReviewDataset, imdb_collate_fn

############

# Since we are running the script sort of ad-hoc, update parameters
# here

current_rnn_model_to_viz = 'coupled'
PADDING_TOKEN = 0

if current_rnn_model_to_viz == 'gru':
  CKPT_VOCABULARY_SIZE = 82
  CKPT_EMBEDDING_DIM = 256
  CKPT_HIDDEN_SIZE = 128
  model_checkpoint_path = 'data/war_and_peace_model_checkpoint.pt'
  loaded_model_checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
  vocab = loaded_model_checkpoint['vocabulary']
  model_checkpoint = loaded_model_checkpoint['model']
  dataset = VisualizeWarAndPeaceDataset(vocab)
else:
  dataset = IMDBReviewDataset(csv_path='data/imdb_train.csv',
                                  vocab_min_count=100,
                                  vocab_max_size=20000,
                                  review_max_length=200)
  vocab = dataset.get_vocabulary()
  CKPT_VOCABULARY_SIZE = len(vocab)
  CKPT_EMBEDDING_DIM = 128
  CKPT_HIDDEN_SIZE = 100
  model_checkpoint_path = 'experiments/{}1_emb_128.h_100/best_model.pt'.format(current_rnn_model_to_viz)
  model_checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


##############

class VisualizeInternalGates(nn.Module):

  def __init__(self, rnn_module):
    super().__init__()
    vocabulary_size = CKPT_VOCABULARY_SIZE
    embedding_dim = CKPT_EMBEDDING_DIM
    hidden_size = CKPT_HIDDEN_SIZE

    self.rnn_module = rnn_module

    self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                  embedding_dim=embedding_dim,
                                  padding_idx=PADDING_TOKEN)
    self.rnn_model = VisualizeGRUCell(input_size=embedding_dim,
                                      hidden_size=hidden_size)
    self.classifier = nn.Linear(hidden_size, vocabulary_size)
    return

  def forward(self, batch_reviews):
    data = self.embedding(batch_reviews)

    state = None
    batch_size, total_steps, _ = data.shape
    internals = []
    for step in range(total_steps):
      next_h, gate_signals = self.rnn_model(data[:, step, :], state)
      internals.append(gate_signals)
      state = next_h

    logits = self.classifier(state)

    internals = list(zip(*internals))

    if (len(internals) == 3):
      outputs = {
          'update_signals': internals[0],
          'reset_signals': internals[1],
          'cell_state_candidates': internals[2],
      }
    else: # (C, f_t, i_t, o_t)
      outputs = {
        'cell_state_candidates': internals[0],
        'forget_signals': internals[1],
        'input_signals': internals[2],
        'output_signals': internals[3]
      }
    return logits, outputs

class VisualizeInternalGatesSentimentClassificaton(nn.Module):

  def __init__(self, rnn_module):
    super().__init__()
    vocabulary_size = CKPT_VOCABULARY_SIZE
    embedding_dim = CKPT_EMBEDDING_DIM
    hidden_size = CKPT_HIDDEN_SIZE
    self.bias = True

    self.rnn_module = rnn_module

    self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                  embedding_dim=embedding_dim,
                                  padding_idx=PADDING_TOKEN)
    self.rnn_model = self.rnn_module(input_size=embedding_dim,
                                      hidden_size=hidden_size)
    self.classifier = nn.Linear(hidden_size, 2)
    return

  def forward(self, batch_reviews):
    data = self.embedding(batch_reviews)

    state = None
    batch_size, total_steps, _ = data.shape
    internals = []
    for step in range(total_steps):
      next_h, gate_signals = self.rnn_model(data[:, step, :], state)
      internals.append(gate_signals)
      state = next_h

    logits = self.classifier(state)

    internals = list(zip(*internals))

    outputs = {
      'cell_state_candidates': internals[0],
      'forget_signals': internals[1],
      'input_signals': internals[2],
      'output_signals': internals[3]
    }
    return logits, outputs


class VisualizeGRUCell(nn.Module):

  def __init__(self, input_size, hidden_size):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size

    self.W_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    self.W_r = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))

    self.reset_parameters()

  def forward(self, x, prev_state):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state

    concat_hx = torch.cat((prev_h, x), dim=1)
    z = torch.sigmoid(F.linear(concat_hx, self.W_z))
    r = torch.sigmoid(F.linear(concat_hx, self.W_r))
    h_tilde = torch.tanh(F.linear(torch.cat((r * prev_h, x), dim=1), self.W))
    next_h = (1 - z) * prev_h + z * h_tilde
    return next_h, (z, r, h_tilde)

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

class VisualizeLSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias
    self.W_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))  
    self.W_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    self.W_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    self.W_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))

    self.b_f = nn.Parameter(torch.Tensor(hidden_size))
    self.b_i = nn.Parameter(torch.Tensor(hidden_size))
    self.b_o = nn.Parameter(torch.Tensor(hidden_size))
    self.b_c = nn.Parameter(torch.Tensor(hidden_size))
    self.reset_parameters()

  def forward(self, x, prev_state):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
      prev_c = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state[0]
      prev_c = prev_state[1]

  
    concat_hx = torch.cat((prev_h, x), dim=1)
    f_t = torch.sigmoid(F.linear(concat_hx, self.W_f, self.b_f))
    i_t = torch.sigmoid(F.linear(concat_hx, self.W_i, self.b_i))
    C_tilde = torch.tanh(F.linear(concat_hx, self.W_c, self.b_c))
    o_t = torch.sigmoid(F.linear(concat_hx, self.W_o, self.b_o))
    
    C = f_t * prev_c + i_t * C_tilde
    h = o_t * torch.tanh(C)
    return h, (C, f_t, i_t, o_t)

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

class VisualizePeepholedLSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
  
  
  
    cell_state_size = hidden_size 
    self.bias = bias

    self.W_f = nn.Parameter(torch.Tensor(hidden_size, cell_state_size + hidden_size + input_size))
    self.W_i = nn.Parameter(torch.Tensor(hidden_size, cell_state_size + hidden_size + input_size))
    self.W_o = nn.Parameter(torch.Tensor(hidden_size, cell_state_size + hidden_size + input_size))
    self.W_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))

    self.b_f = nn.Parameter(torch.Tensor(hidden_size))
    self.b_i = nn.Parameter(torch.Tensor(hidden_size))
    self.b_o = nn.Parameter(torch.Tensor(hidden_size))
    self.b_c = nn.Parameter(torch.Tensor(hidden_size))
    return

  def forward(self, x, prev_state):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
      prev_c = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state[0]
      prev_c = prev_state[1]

  
    concat_hx = torch.cat((prev_h, x), dim=1)
  
    concat_chx = torch.cat((prev_c, prev_h, x), dim=1)

  
  
    f_t = torch.sigmoid(F.linear(concat_chx, self.W_f, self.b_f))
    i_t = torch.sigmoid(F.linear(concat_chx, self.W_i, self.b_i))
    C_tilde = torch.tanh(F.linear(concat_hx, self.W_c, self.b_c))
    o_t = torch.sigmoid(F.linear(concat_chx, self.W_o, self.b_o))
    
    C = f_t * prev_c + i_t * C_tilde
    h = o_t * torch.tanh(C)
    return h, (C, f_t, i_t, o_t)

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

class VisualizeCoupledLSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias
    
    self.W_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
  

    self.W_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    self.W_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))

    self.b_f = nn.Parameter(torch.Tensor(hidden_size))
    self.b_o = nn.Parameter(torch.Tensor(hidden_size))
    self.b_c = nn.Parameter(torch.Tensor(hidden_size))
    self.reset_parameters()

  def forward(self, x, prev_state):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
      prev_c = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state[0]
      prev_c = prev_state[1]

    concat_hx = torch.cat((prev_h, x), dim=1)
    f_t = torch.sigmoid(F.linear(concat_hx, self.W_f, self.b_f))
    i_t = 1 - f_t
    C_tilde = torch.tanh(F.linear(concat_hx, self.W_c, self.b_c))
    o_t = torch.sigmoid(F.linear(concat_hx, self.W_o, self.b_o))
    
    C = f_t * prev_c + i_t * C_tilde
    h = o_t * torch.tanh(C)
    return h, (C, f_t, i_t, o_t)

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

class VisualizeWarAndPeaceDataset(Dataset):

  def __init__(self, vocabulary):
    self.vocabulary = vocabulary

    # Hardcode the parameters to match the provided checkpoint
    txt_path = 'data/war_and_peace_visualize.txt'

    with open(txt_path, 'rb') as fp:
      raw_text = fp.read().strip().decode(encoding='utf-8')

    self.data = raw_text.split('\n')

    self.char2index = {x: i for (i, x) in enumerate(self.vocabulary)}
    self.index2char = {i: x for (i, x) in enumerate(self.vocabulary)}

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return np.array([self.char2index[x] for x in self.data[index]]), -1

  def convert_to_chars(self, sequence):
    if isinstance(sequence, torch.Tensor):
      sequence = sequence.squeeze(0).detach().numpy().tolist()
    return [self.index2char[x] for x in sequence]


def visualize_internals(sequence_id,
                        sequence,
                        gate_name,
                        states,
                        saving_dir='visualize/'):
  states = torch.cat(states, dim=0).detach().numpy().T
  hidden_size, time_stamps = states.shape
  fig, ax = plt.subplots(figsize=(time_stamps / 5, hidden_size / 5))

  if gate_name in ['update_signals', 'reset_signals']:
    vmin = 0
  elif gate_name == 'cell_state_candidates':
    vmin = -1
  else:
    vmin = 0

  for tick in ax.get_xticklabels():
    tick.set_rotation(80)

  sns.heatmap(states,
              cbar=False,
              square=True,
              linewidth=0.05,
              xticklabels=sequence,
              yticklabels=False,
              vmin=vmin,
              vmax=1,
              cmap='bwr',
              ax=ax)

  plt.xlabel('Sequence')
  plt.ylabel('Hidden Cells')

  locs, labels = plt.xticks()
  ax.xaxis.set_ticks_position('top')
  ax.xaxis.set_ticklabels(labels, rotation=90)

  plt.tight_layout()
  os.makedirs(saving_dir, exist_ok=True)
  plt.savefig(
      os.path.join(saving_dir,
                   'S%02d_' % sequence_id + gate_name.lower() + '.png'))
  plt.close()
  return


RNN_MODULES = {
    'gru': VisualizeGRUCell,
    'lstm': VisualizeLSTMCell,
    'peephole': VisualizePeepholedLSTMCell,
    'coupled': VisualizeCoupledLSTMCell
}


def imdb_visualizer():
  # Wasnt sure if you wanted us to keep this one function so I 
  # put the code in here
  data_loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=imdb_collate_fn)
  model = VisualizeInternalGatesSentimentClassificaton(RNN_MODULES[current_rnn_model_to_viz])
  model.load_state_dict(model_checkpoint)
  sequence, _ = dataset[0]
  sentence = [dataset.index2word.get(w, dataset.oov_token_id) for w in sequence]
  _, gates = model(torch.LongTensor(sequence).view(-1, 1))
  for gate_name in gates.keys():
    visualize_internals(0, sentence, gate_name, gates[gate_name])



def war_and_peace_visualizer():
  if current_rnn_model_to_viz == 'gru':
    data_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=8)
  else:
    imdb_visualizer()
    return

  model = VisualizeInternalGates(RNN_MODULES[current_rnn_model_to_viz])
  model.load_state_dict(model_checkpoint)
  model.eval()
  for step, data in enumerate(data_loader):
    sequences = data[0]
    total_step = len(data_loader) + step
    sequences = sequences.to(device)
    logits, gates = model(sequences)
    for gate_name in gates.keys():
      visualize_internals(step, dataset.convert_to_chars(sequences), gate_name, gates[gate_name])


def main(unused_argvs):
  war_and_peace_visualizer()


if __name__ == '__main__':
  app.run(main)
