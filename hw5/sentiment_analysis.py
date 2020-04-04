import collections
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import IMDBReviewDataset, imdb_collate_fn
from rnn_modules import CoupledLSTMCell, GRUCell, LSTMCell, PeepholedLSTMCell

FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 0, 'Weight decay (L2 regularization).')
flags.DEFINE_integer('batch_size', 4096, 'Number of examples per batch.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs for training.')
flags.DEFINE_string('experiment_name', 'exp', 'Defines experiment name.')
flags.DEFINE_string('model_checkpoint', '',
                    'Specifies the checkpont for analyzing.')

flags.DEFINE_integer('embedding_dim', 128,
                     'Dimensionality for word embeddings.')
flags.DEFINE_integer('hidden_size', 100, 'Dimensionality for recurrent neuron.')
flags.DEFINE_integer('review_max_length', 200,
                     'Truncates reviews that beyond the threshold.')
flags.DEFINE_integer('vocabulary_min_count', 100,
                     'Preserving words that occur more than this times.')
flags.DEFINE_integer('vocabulary_max_size', 20000,
                     'Preserving words that occur more than this times.')
flags.DEFINE_enum('rnn_module', 'gru',
                  ['lstm', 'gru', 'peepholed_lstm', 'coupled_lstm'],
                  'Specifies the recurrent module in the RNN.')

PADDING_TOKEN = 0
GRADIENT_CLIP_NORM = 1.0

RNN_MODULES = {
    'gru': GRUCell,
    'lstm': LSTMCell,
    'peepholed_lstm': PeepholedLSTMCell,
    'coupled_lstm': CoupledLSTMCell,
}


class SentimentClassification(nn.Module):

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

    self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                  embedding_dim=embedding_dim,
                                  padding_idx=PADDING_TOKEN)
    self.rnn_model = self.rnn_module(input_size=embedding_dim,
                                     hidden_size=hidden_size,
                                     bias=bias)
    self.classifier = nn.Linear(hidden_size, 2)
    return

  def forward(self, batch_reviews, batch_lengths):
    data = self.embedding(batch_reviews)

    state = None
    batch_size, total_steps, _ = data.shape
    full_outputs = []
    for step in range(total_steps):
      next_state = self.rnn_model(data[:, step, :], state)
      if isinstance(next_state, tuple):
        h, c = next_state
        full_outputs.append(h)
      else:
        full_outputs.append(next_state)
      state = next_state

    full_outputs = torch.stack(full_outputs, dim=1)
    outputs = full_outputs[torch.arange(batch_size), batch_lengths - 1, :]
    logits = self.classifier(outputs)
    return logits


def imdb_trainer():
  train_dataset = IMDBReviewDataset(csv_path='data/imdb_train.csv',
                                    vocab_min_count=FLAGS.vocabulary_min_count,
                                    vocab_max_size=FLAGS.vocabulary_max_size,
                                    review_max_length=FLAGS.review_max_length)
  train_dataset.print_statistics()
  train_loader = DataLoader(train_dataset,
                            batch_size=FLAGS.batch_size,
                            shuffle=True,
                            num_workers=8,
                            collate_fn=imdb_collate_fn)
  vocabulary = train_dataset.get_vocabulary()

  # Validation dataset should use the same vocabulary as the training set.
  val_dataset = IMDBReviewDataset(csv_path='data/imdb_test.csv',
                                  vocabulary=vocabulary,
                                  review_max_length=FLAGS.review_max_length)
  val_dataset.print_statistics()
  val_loader = DataLoader(val_dataset,
                          batch_size=FLAGS.batch_size,
                          shuffle=False,
                          num_workers=8,
                          collate_fn=imdb_collate_fn)

  best_model = None
  best_acc = 0.0

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  experiment_name = 'experiments/{}_emb_{}.h_{}'.format(FLAGS.experiment_name,
                                                        FLAGS.embedding_dim,
                                                        FLAGS.hidden_size)

  rnn_modules = {
      'gru': GRUCell,
      'lstm': LSTMCell,
      'peepholed_lstm': PeepholedLSTMCell,
      'coupled_lstm': CoupledLSTMCell,
  }

  os.makedirs(experiment_name, exist_ok=True)
  writer = SummaryWriter(log_dir=experiment_name)

  model = SentimentClassification(vocabulary_size=len(vocabulary),
                                  embedding_dim=FLAGS.embedding_dim,
                                  rnn_module=rnn_modules[FLAGS.rnn_module],
                                  hidden_size=FLAGS.hidden_size)
  model.to(device)

  print('Model Architecture:\n%s' % model)

  criterion = nn.CrossEntropyLoss(reduction='mean')
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=FLAGS.learning_rate,
                               weight_decay=FLAGS.weight_decay)

  try:
    for epoch in range(FLAGS.epochs):
      for phase in ('train', 'eval'):
        if phase == 'train':
          model.train()
          dataset = train_dataset
          data_loader = train_loader
        else:
          model.eval()
          dataset = val_dataset
          data_loader = val_loader

        running_loss = 0.0
        running_corrects = 0

        for step, (reviews, lengths, labels) in tqdm(enumerate(data_loader)):
          reviews = reviews.to(device)
          lengths = lengths.to(device)
          labels = labels.to(device)

          optimizer.zero_grad()

          with torch.set_grad_enabled(phase == 'train'):
            outputs = model(reviews, lengths)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            if phase == 'train':
              loss.backward()

              # RNN model is easily getting exploded gradients, thus we perform
              # gradients clipping to mitigate this issue.
              nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
              optimizer.step()

              writer.add_scalar('Loss/{}'.format(phase), loss.item(),
                                epoch * len(data_loader) + step)

          running_loss += loss.item() * reviews.size(0)
          running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)
        writer.add_scalar('Epoch_Loss/{}'.format(phase), epoch_loss, epoch)
        writer.add_scalar('Epoch_Accuracy/{}'.format(phase), epoch_acc, epoch)
        print('[Epoch %d] %s accuracy: %.4f, loss: %.4f' %
              (epoch + 1, phase, epoch_acc, epoch_loss))

        if phase == 'eval':
          if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, os.path.join(experiment_name,
                                                'best_model.pt'))

  except KeyboardInterrupt:
    pass

  return


def main(unused_argvs):
  imdb_trainer()


if __name__ == '__main__':
  app.run(main)
