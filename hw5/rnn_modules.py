import torch.nn as nn
import torch.nn.functional as F
import torch


class GRUCell(nn.Module):
  """Implementation of GRU cell from https://arxiv.org/pdf/1406.1078.pdf."""

  def __init__(self, input_size, hidden_size, bias=False):
    # https://www.youtube.com/watch?v=8HyCNIVRbSU&feature=emb_title
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    # Learnable weights and bias for `update gate`
    self.W_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_z = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_z', None)

    # Learnable weights and bias for `reset gate`
    self.W_r = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_r = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_r', None)

    # Learnable weights and bias for `output gate`
    self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b', None)

    self.reset_parameters()

  def forward(self, x, prev_state):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state

    concat_hx = torch.cat((prev_h, x), dim=1)
    z = torch.sigmoid(F.linear(concat_hx, self.W_z, self.b_z))
    r = torch.sigmoid(F.linear(concat_hx, self.W_r, self.b_r))
    h_tilde = torch.tanh(
        F.linear(torch.cat((r * prev_h, x), dim=1), self.W, self.b))
    next_h = (1 - z) * prev_h + z * h_tilde
    return next_h

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}, bias={}'.format(
        self.input_size, self.hidden_size, self.bias is not True)

  def count_parameters(self):
    print('Total Parameters: %d' %
          sum(p.numel() for p in self.parameters() if p.requires_grad))
    return


class LSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    # Learnable weights and bias for `forget gate`
    self.W_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_f = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_f', None)

    # Learnable weights and bias for `input gate`
    self.W_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_i = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_i', None)

    # Learnable weights and bias for `output gate`
    self.W_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_o = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_o', None)

    # Learnable weights and bias for `g gate`, or as colah says, C_t tilde
    self.W_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_c = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_c', None)

    self.reset_parameters()

  def forward(self, x, prev_state):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
      prev_c = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state[0]
      prev_c = prev_state[1]

    # We can stack prev_h and x on top of each other, and apply matrix multiplies to them
    concat_hx = torch.cat((prev_h, x), dim=1)

    # Formulas are from https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    # The only way to understand what this actually does is read that
    f_t = torch.sigmoid(F.linear(concat_hx, self.W_f, self.b_f))
    i_t = torch.sigmoid(F.linear(concat_hx, self.W_i, self.b_i))
    C_tilde = torch.tanh(F.linear(concat_hx, self.W_c, self.b_c))
    o_t = torch.sigmoid(F.linear(concat_hx, self.W_o, self.b_o))
    
    C = f_t * prev_c + i_t * C_tilde # next cell state
    h = o_t * torch.tanh(C)
    return h, C

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}, bias={}'.format(
        self.input_size, self.hidden_size, self.bias is not True)

  def count_parameters(self):
    print('Total Parameters: %d' %
          sum(p.numel() for p in self.parameters() if p.requires_grad))
    return


class PeepholedLSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    # cell state size is the same as hidden, and in peephole we
    # stack prev cell state on the prev hidden state and input,
    # which changes the size of our network
    cell_state_size = hidden_size 
    self.bias = bias

    self.W_f = nn.Parameter(torch.Tensor(hidden_size, cell_state_size + hidden_size + input_size))
    if bias:
      self.b_f = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_f', None)

    self.W_i = nn.Parameter(torch.Tensor(hidden_size, cell_state_size + hidden_size + input_size))
    if bias:
      self.b_i = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_i', None)

    self.W_o = nn.Parameter(torch.Tensor(hidden_size, cell_state_size + hidden_size + input_size))
    if bias:
      self.b_o = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_o', None)

    self.W_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_c = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_c', None)

    return

  def forward(self, x, prev_state):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
      prev_c = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state[0]
      prev_c = prev_state[1]

    # We can stack prev_h and x on top of each other, and apply matrix multiplies to them
    concat_hx = torch.cat((prev_h, x), dim=1)
    # The "peephole" is that all gates have access to C
    concat_chx = torch.cat((prev_c, prev_h, x), dim=1)

    # Formulas are from https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    # The only way to understand what this actually does is read that
    f_t = torch.sigmoid(F.linear(concat_chx, self.W_f, self.b_f))
    i_t = torch.sigmoid(F.linear(concat_chx, self.W_i, self.b_i))
    C_tilde = torch.tanh(F.linear(concat_hx, self.W_c, self.b_c))
    o_t = torch.sigmoid(F.linear(concat_chx, self.W_o, self.b_o))
    
    C = f_t * prev_c + i_t * C_tilde # next cell state
    h = o_t * torch.tanh(C)
    return h, C

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}, bias={}'.format(
        self.input_size, self.hidden_size, self.bias is not True)

  def count_parameters(self):
    print('Total Parameters: %d' %
          sum(p.numel() for p in self.parameters() if p.requires_grad))
    return


class CoupledLSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias
    
    self.W_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_f = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_f', None)

    # Input gate is 'coupled' to forget gate. so i = 1 - f!

    self.W_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_o = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_o', None)

    self.W_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_c = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_c', None)

    self.reset_parameters()

  def forward(self, x, prev_state):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
      prev_c = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state[0]
      prev_c = prev_state[1]

    # We can stack prev_h and x on top of each other, and apply matrix multiplies to them
    concat_hx = torch.cat((prev_h, x), dim=1)

    # Formulas are from https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    # The only way to understand what this actually does is read that
    f_t = torch.sigmoid(F.linear(concat_hx, self.W_f, self.b_f))
    i_t = 1 - f_t
    C_tilde = torch.tanh(F.linear(concat_hx, self.W_c, self.b_c))
    o_t = torch.sigmoid(F.linear(concat_hx, self.W_o, self.b_o))
    
    C = f_t * prev_c + i_t * C_tilde # next cell state
    h = o_t * torch.tanh(C)
    return h, C

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}, bias={}'.format(
        self.input_size, self.hidden_size, self.bias is not True)

  def count_parameters(self):
    print('Total Parameters: %d' %
          sum(p.numel() for p in self.parameters() if p.requires_grad))
    return


RNN_MODULES = {
    'gru': GRUCell,
    'lstm': LSTMCell,
    'peepholed_lstm': PeepholedLSTMCell,
    'coupled_lstm': CoupledLSTMCell
}

if __name__ == '__main__':
  for name, lstm in RNN_MODULES.items():
    l = lstm(128, 100, bias=True)
    print(name.upper())
    l.count_parameters()
