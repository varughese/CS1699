import torch
import numpy as np
from os import path as ospath
from skimage import io
import torch.nn as nn
import os
from pathlib import Path
import wandb

wandb.init(project="cs1699-hw3")

class CifarDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir):
    """Initializes a dataset containing images and labels."""
    super().__init__()
    root_dir = Path(root_dir)
    paths_and_labels = [[str(p), self.image_path_to_number(p)] for p in root_dir.rglob("*.png")]
    self.data = np.array(paths_and_labels)

  def image_path_to_number(self, image_path):
    # cifar10_train/cat/cat_00840.png for example, we want to 
    # turn cat 
    LABEL_TO_NUMBER = {
      "airplane": 0,
      "automobile": 1,
      "bird": 2,
      "cat": 3,
      "deer": 4,
      "dog": 5,
      "frog": 6,
      "horse": 7,
      "ship": 8,
      "truck": 9
    }
    return LABEL_TO_NUMBER[image_path.parts[-2]]

  def __len__(self):
    """Returns the size of the dataset."""
    return len(self.data)

  def __getitem__(self, index):
    """Returns the index-th data item of the dataset."""
    [img_path, img_label] = self.data[index]
    return (io.imread(img_path), int(img_label)) 

# Define our model (3-layer MLP)
class MultilayerPerceptron(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()
    self.input_size = input_size
    self.fc1 = nn.Linear(input_size, hidden_size) 
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)
  
  def forward(self, x):
    dropout = nn.Dropout(p=0.3)
    out = self.fc1(x.reshape(-1, self.input_size))
    out = dropout(out)
    out = self.relu(out)
    out = self.fc2(out)
    return out

def training(training_data_path, device, model, criterion, optimizer):
  train_dataset = CifarDataset(training_data_path)
  train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True)

  # Set the model into `training` mode, because certain operators 
  # will perform differently during training and evaluation 
  # (e.g. dropout and batch normalization)
  model.train() 
  
  for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_dataloader):  
      # Move tensors to the configured device
      images = images.to(device)
      labels = labels.to(device)
    
      # Forward pass
      outputs = model(images.float())
      loss = criterion(outputs, labels)

      # Backward and optimize
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      wandb.log({'epoch': epoch, 'loss': loss.item()})

      if (i + 1) % 10 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, NUM_EPOCHS, i + 1, len(train_dataloader), loss.item()))
        

def evaluation(test_data_path, device, model):
  test_dataset = CifarDataset(test_data_path)
  test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True)
  # Test the model
  # In test phase, we don't need to compute gradients (for memory efficiency)
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataloader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images.float())
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))



TRAIN_DIRECTORY_PATH = "cifar10/cifar10_train"
TEST_DIRECTORY_PATH = "cifar10/cifar10_test"
INPUT_SIZE = 32*32*3 # 32 x 32 x RGB Images
NUM_CLASSES = 10

BATCH_SIZE = 50
HIDDEN_SIZE = 500
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

device = 'cpu'
model = MultilayerPerceptron(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

wandb.init(config={"epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE})
wandb.watch(model)

training(TRAIN_DIRECTORY_PATH, device, model, criterion, optimizer)
evaluation(TEST_DIRECTORY_PATH, device, model)

torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))