
- initial condition: all zero? many are studying on this subject. some use a normal distribution

- batch size: break down data for a better train that one-by-one and all-at-once
  - at each epoch each batch is different than the one in previous epoch because of randomization.

- epoch: data seen at least once
  - good number of epoch?
    - early stopping: from a certain point there aren't any significant change in loss function
    - check points: choosing best parameters (based on validation data and not train data)
    - plotting cost function vs number of epochs: when there are not much change or overfitting
      is happening we stop

- Hyperparameter tuning
  - classic methods (not so good)
    - grid search
    - random search
  - Bayesian optimization
    - based on a little bit of history

- DataLoader makes iterable object out of DataSets
  - the iterable (`getitem` dunder) will return both _data_ and the corresponding _label_
  - each time it will return 32 data and 32 label because our batch size is 32

  ```
  data, label = next(iter(train_dloader))
  print("Data:\t", data.size())
  print("Label:\t", label.size())
  ```

  ```
  Data:	 torch.Size([10, 1, 28, 28])
  Label:	 torch.Size([10])
  ```

  - data and label are each torch tensor and have all properties and methods of a torch tensor (e.g.
    size)
  - `[10, 1, 28]`:
    - 10: size of the data tensor
    - 1: channel
    - 28,28: 28x28 (size of the window (pixels))


- `class NN(nn.Module):` This is where we build our neural network and its layers
  - each layer IS A `nn.Module` itself
  - there are bunch of built-in modules in torch:
    - Linear
    - SVM
    - ...
  - combination of each of these models makes a custom model that we use
