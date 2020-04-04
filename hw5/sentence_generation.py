import collections
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from torch.distributions import categorical
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import ShakespeareDataset
from rnn_modules import CoupledLSTMCell, GRUCell, LSTMCell, PeepholedLSTMCell

FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 0, 'Weight decay (L2 regularization).')
flags.DEFINE_integer('batch_size', 2048, 'Number of examples per batch.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs for training.')
flags.DEFINE_string('experiment_name', 'exp', 'Defines experiment name.')
flags.DEFINE_string('model_checkpoint', '',
                    'Specifies the checkpont for analyzing.')

flags.DEFINE_integer('embedding_dim', 50, 'Dimensionality for word embeddings.')
flags.DEFINE_integer('hidden_size', 50, 'Dimensionality for recurrent neuron.')

flags.DEFINE_enum('rnn_module', 'gru',
                  ['lstm', 'gru', 'peepholed_lstm', 'coupled_lstm'],
                  'Specifies the recurrent module in the RNN.')

flags.DEFINE_integer('history_length', 100,
                     'Number of characters to check for predicting next char.')
flags.DEFINE_integer('generation_length', 100,
                     'Number of characters to generate.')
flags.DEFINE_string('start_string', 'R', 'Start string for generation.')
flags.DEFINE_enum('task_type', 'training', ['training', 'generation'],
                  'Specifies the type of the task.')

PADDING_TOKEN = 0

RNN_MODULES = {
    'gru': GRUCell,
    'lstm': LSTMCell,
    'peepholed_lstm': PeepholedLSTMCell,
    'coupled_lstm': CoupledLSTMCell,
}


class SentenceGeneration(nn.Module):

  def __init__(self,
               vocabulary_size,
               embedding_dim,
               rnn_module,
               hidden_size,
               bias=False):
    super().__init__()
    self.vocabulary_size = vocabulary_size
    self.rnn_module = rnn_module
    self.embedding_dim = embedding_dim
    self.hidden_size = hidden_size
    self.bias = bias

    #####################################################################
    # Implement here following the given signature                      #
    raise NotImplementedError
    #####################################################################

    return

  def forward(self, history, state=None):
    """Predicts next character.

    Use a for-loop to iteratively process all charcters in history then predicts
    the probability distrution over full vocabulary for next character.

    Note that when state is set to None, you should initialize the initial state
    as all zeros; otherwise when state is provided, the model should continue
    from state. This will be very useful for generating new sentences.
    
    Args:
      history: Iterable of character ids.
      state: Optional, the cell state for RNN. If not provided the cell state
        will be initialized as all zeros.

    Returns:
      logits: Predicted logits (before softmax) over vocabulary.
      state: Current state, useful for continuous inference.
    """
    #####################################################################
    # Implement here following the given signature                      #
    raise NotImplementedError
    # Placeholder, you need to override these two variables.
    logits, state = None, None
    #####################################################################
    return logits, state

  def reset_parameters(self):
    with torch.no_grad:
      for param in self.parameters():
        param.reset_parameters()
    return


def shakespeare_trainer():
  train_dataset = ShakespeareDataset(txt_path='data/shakespeare.txt',
                                     history_length=FLAGS.history_length)
  train_loader = DataLoader(train_dataset,
                            batch_size=FLAGS.batch_size,
                            shuffle=True,
                            num_workers=8)
  vocabulary = train_dataset.get_vocabulary()

  best_model = None
  best_loss = 0.0

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  experiment_name = 'experiments/{}_emb_{}.h_{}'.format(FLAGS.experiment_name,
                                                        FLAGS.embedding_dim,
                                                        FLAGS.hidden_size)

  os.makedirs(experiment_name, exist_ok=True)
  writer = SummaryWriter(log_dir=experiment_name)

  model = SentenceGeneration(vocabulary_size=len(vocabulary),
                             embedding_dim=FLAGS.embedding_dim,
                             rnn_module=RNN_MODULES[FLAGS.rnn_module],
                             hidden_size=FLAGS.hidden_size)
  model.to(device)

  print('Model Architecture:\n%s' % model)

  criterion = nn.CrossEntropyLoss(reduction='mean')
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=FLAGS.learning_rate,
                               weight_decay=FLAGS.weight_decay)

  try:
    for epoch in range(FLAGS.epochs):
      model.train()
      dataset = train_dataset
      data_loader = train_loader

      progress_bar = tqdm(enumerate(data_loader))
      for step, (sequences, labels) in progress_bar:
        total_step = epoch * len(data_loader) + step
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs, _ = model(sequences)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        corrects = torch.sum(preds == labels.data)

        loss.backward()
        optimizer.step()

        writer.add_scalar('loss', loss.item(), total_step)
        writer.add_scalar('accuracy', corrects.item() / len(labels), total_step)
        progress_bar.set_description(
            'Loss: %.4f, Accuracy: %.4f' %
            (loss.item(), corrects.item() / len(labels)))

      model_copy = copy.deepcopy(model.state_dict())
      torch.save({
          'model': model_copy,
          'vocabulary': vocabulary
      }, os.path.join(experiment_name, 'model_epoch_%d.pt' % (epoch + 1)))

  except KeyboardInterrupt:
    pass

  final_model = copy.deepcopy(model.state_dict())
  torch.save({
      'model': final_model,
      'vocabulary': vocabulary
  }, os.path.join(experiment_name, 'best_model.pt'))

  return


def sample_next_char_id(predicted_logits):
  next_char_id = categorical.Categorical(logits=predicted_logits).sample()
  return next_char_id


def shakespeare_writer():
  """Generates new sentences using trained language model."""
  start_string = FLAGS.start_string
  device = 'cpu'

  state_dict = torch.load(FLAGS.model_checkpoint)
  vocabulary = state_dict['vocabulary']

  char2index = {x: i for (i, x) in enumerate(vocabulary)}
  index2char = {i: x for (i, x) in enumerate(vocabulary)}

  inputs = torch.tensor([char2index[x] for x in start_string])
  inputs = inputs.view(1, -1)

  model = SentenceGeneration(vocabulary_size=len(vocabulary),
                             embedding_dim=FLAGS.embedding_dim,
                             rnn_module=RNN_MODULES[FLAGS.rnn_module],
                             hidden_size=FLAGS.hidden_size)

  model.load_state_dict(state_dict['model'])
  model.eval()

  generated_chars = []
  #####################################################################
  # Implement here for generating new sentence                        #
  # Specifically, you need to iterate through the history and predict #
  # next character; then you could take the predicted history as part #
  # of history then repeat the process. The generation should be      #
  # repeated for FLAGS.generation_length times.
  raise NotImplementedError
  #####################################################################

  return start_string + ''.join(generated_chars)


def main(unused_argvs):
  if FLAGS.task_type == 'training':
    shakespeare_trainer()
  elif FLAGS.task_type == 'generation':
    shakespeare_writer()


if __name__ == '__main__':
  app.run(main)
