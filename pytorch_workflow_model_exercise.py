# -*- coding: utf-8 -*-
"""pytorch_workflow_model_exercise.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1goJAymCkKUDVZ8CuT_QoYz04YK5vdMff
"""

# import libraries
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# check torch version
torch.__version__

# Setup agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# data
weight = 0.7
bias = 0.3

# create range of values
start = 0
end = 1
step = 0.02

# create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1)

y = weight * X + bias

# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

# create function for plot
def plot_preds(train_data=X_train,
               train_labels=y_train,
               test_data=X_test,
               test_labels=y_test,
               predictions=None):
  # plot train data
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data") # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
  # plot test data
  plt.scatter(test_data, test_labels, c="g", s=4, label="Test Data")

  # check if 'predictions' exist
  if predictions is not None:
    # plot the predictions
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # show the legend
  plt.legend(prop={"size": 14})

plot_preds()

# Create our model
class LinearRegressionModelExercise(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_layer = nn.Linear(in_features=1,
                                   out_features=1)

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)

# set the manual seed
torch.manual_seed(42)

# create instance of our model
model = LinearRegressionModelExercise()

model, model.state_dict()

# check the model current device
next(model.parameters()).device

# Set the model to use target device
model.to(device)
next(model.parameters()).device

# for training our model we need 4 things on our hand: 1. loss function 2. optimizer 3. training loop and 4. testing loop
# 1. loss function
loss_fnc = nn.L1Loss()

# 2. optimizer
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.01)

# 3. training loop
torch.manual_seed(42)

epochs=200

# put the data to the target device (device agnosting code for data)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
  model.train()

  # 1. Forward pass
  y_pred = model(X_train)

  # 2. Calculate the loss
  loss = loss_fnc(y_pred, y_train)

  # 3. Optimizer zeor grad
  optimizer.zero_grad()

  # 4. backpropagation
  loss.backward()

  # 5. optimizer step
  optimizer.step()


  ### Testing
  model.eval()

  with torch.inference_mode():
    test_pred = model(X_test)

    # Calculate the loss
    test_loss = loss_fnc(test_pred, y_test)

  # print out the result
  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss} | Test Loss | {test_loss}")

weight, bias

model.state_dict()

# make predictions
model.eval()
with torch.inference_mode():
  y_preds = model(X_test)
y_preds

plot_preds(predictions=y_preds.cpu())

"""### 6.5 Saving and loading trained model\

There are three main methods you should about for saving and loading models in PyTorch.

1. `torch.save()` - allows you to save a PyTorch object in [Python's pickle](https://docs.python.org/3/library/pickle.html) format
2. `torch.load()` - allows you to load a PyTorch object
3. `torch.nn.Module.load_state-dict()` - this allows to load a model's saved state dictionary
"""

from pathlib import Path

# Create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# create a model save path
MODEL_NAME = "pytorch_workflow_model_exercise.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f= MODEL_SAVE_PATH)
