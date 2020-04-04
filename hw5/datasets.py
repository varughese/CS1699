import collections
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PADDING_TOKEN = 0


class IMDBReviewDataset(Dataset):

  def __init__(self,
               csv_path,
               vocabulary=None,
               vocab_min_count=10,
               vocab_max_size=None,
               review_max_length=200):
    self.csv_path = csv_path
    self.vocab_min_count = vocab_min_count
    self.vocab_max_size = vocab_max_size
    self.review_max_length = review_max_length - 2

    self.data = []
    with open(csv_path, 'r') as fp:
      reader = csv.DictReader(fp, delimiter=',')
      for row in tqdm(reader):
        self.data.append((row['review'].split(' ')[:review_max_length],
                          int(row['sentiment'] == 'positive')))

    if vocabulary is not None:
      print('Using external vocabulary - vocab-related configs ignored.')
      self.vocabulary = vocabulary
    else:
      self.vocabulary = self._build_vocabulary()

    self.word2index = {w: i for (i, w) in enumerate(self.vocabulary)}
    self.index2word = {i: w for (i, w) in enumerate(self.vocabulary)}
    self.oov_token_id = self.word2index['OOV_TOKEN']
    self.pad_token_id = self.word2index['PAD_TOKEN']

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    review, label = self.data[index]
    review = ['BEGIN_TOKEN'] + review + ['END_TOKEN']
    token_ids = [self.word2index.get(w, self.oov_token_id) for w in review]
    return token_ids, label

  def _build_vocabulary(self):
    special_tokens = ['PAD_TOKEN', 'BEGIN_TOKEN', 'OOV_TOKEN', 'END_TOKEN']

    counter = collections.Counter()
    for review, _ in self.data:
      counter.update(review)

    vocab = counter.most_common(self.vocab_max_size - 4)
    if self.vocab_min_count is not None:
      vocab_tokens = [w for (w, c) in vocab if c >= self.vocab_min_count]
    else:
      vocab_tokens, _ = zip(vocab)

    return special_tokens + vocab_tokens

  def get_vocabulary(self):
    return self.vocabulary

  def print_statistics(self):
    reviews, labels = zip(*self.data)
    lengths = [len(x) for x in reviews]
    positive = np.sum(labels)
    negative = len(labels) - positive
    print('Total instances: %d, positive: %d, negative: %d' %
          (len(self.data), positive, negative))
    print('Review lengths: max: %d, min: %d, mean: %d, median: %d' %
          (max(lengths), min(lengths), np.mean(lengths), np.median(lengths)))
    print('Vocabulary size: %d' % len(self.vocabulary))
    return


def imdb_collate_fn(batch_data, padding_token_id=PADDING_TOKEN):
  """Padding variable-length sequences."""
  batch_tokens, batch_labels = zip(*batch_data)
  lengths = [len(x) for x in batch_tokens]
  max_length = max(lengths)

  padded_tokens = []
  for tokens, length in zip(batch_tokens, lengths):
    padded_tokens.append(tokens + [padding_token_id] * (max_length - length))

  padded_tokens = torch.tensor(padded_tokens, dtype=torch.int64)
  lengths = torch.tensor(lengths, dtype=torch.int64)
  labels = torch.tensor(batch_labels, dtype=torch.int64)

  return padded_tokens, lengths, labels


class ShakespeareDataset(Dataset):

  def __init__(self, txt_path, history_length):
    self.txt_path = txt_path
    self.history_length = history_length

    with open(txt_path, 'rb') as fp:
      raw_text = fp.read().strip().decode(encoding='utf-8')

    self.vocab = sorted(set(raw_text))
    self.char2index = {x: i for (i, x) in enumerate(self.vocab)}
    self.index2char = {i: x for (i, x) in enumerate(self.vocab)}

    self.data = [(raw_text[i:i + history_length], raw_text[i + history_length])
                 for i in range(len(raw_text) - history_length)]
    return

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    history, label = self.data[index]
    history = np.array([self.char2index[x] for x in history])
    label = self.char2index[label]
    return history, label

  def get_vocabulary(self):
    return self.vocab
