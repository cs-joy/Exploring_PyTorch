
# PyTorch Workflow

Let's explore an example PyTorch end-to-end workflow
"""

what_we_are_covering = {
    1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inferences)",
    5: "saving and loading a model",
    6: "putting it all together"
}

what_we_are_covering

import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks # https://pytorch.org/docs/stable/nn.html

import matplotlib.pyplot as plt

torch.__version__

"""## 1. Data (preparing and loading)

Data can be almost anything... in machine learning

* Excel spreadsheet
* Images of any kind
* Videos (YouTube has lots of data...)
* Audio like songs or podcasts
* DNA
* Text

Machine learning is game of two parts:
1. Get data into a numerical representation
2. Build a model to learn patterns in that numerical representation.

To showcase this, let's create some *known* data using the linear regression formula.

We will use a linear regression formula to make a straight line with *known* **parameters**.
"""

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create a range of numbers
start = 0
end  = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = bias + weight * X # y = a + b*x

X[:10], y[:10]

len(X), len(y)

"""### Splitting data into training and test sets (one of the most important conceptsw in machine learning in general)

Let's create a training and test set with our data.
"""

# Create a train/test split
train_split = int(0.8 * len(X))
#train_split # 40
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

X_train, y_train

"""How might we better visualize our data

This is where the data explorer's motto comes in!

"Visualize, Visualize, Visualize!"
"""

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
  """
  Plot training data, test data and compares predictions.
  """
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")

  # Plot  test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Test Data")

  # Are there predictions?
  if predictions is not None:
    # Plot the predictions if they exist
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14})

plot_predictions();

"""## 2. Build model

Our first PyTorch model

This is very exciting... let's do it

# [Learn from Python OOP](https://realpython.com/python3-object-oriented-programming/)
"""

from torch import nn

# Create a linear regression model class
class LinearRegressionModel(nn.Module): # -> https://pytorch.org/docs/stable/generated/torch.nn.Module.html
  def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=float))