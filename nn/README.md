# CNN Image Classifer

Developed using the Fashion MNST dataset from [Zalando](https://github.com/zalandoresearch/fashion-mnist).

## File Structure

Requirements.txt should be installed with pip in a virtual machine to avoid mismatches in package versions. 

All modifiable code can be found within `if __name__ == "__main__"` section of the relevant script. 

```
nn
 ┣ data
 ┣ models
 ┣ utils
 ┃ ┣ __init__.py
 ┃ ┗ mnist_reader.py
 ┣ README.md
 ┣ model.py
 ┣ pad.ipynb
 ┣ requirements.txt
 ┣ train.py
 ┗ tuning.py
```

## Data Preperation and Loading

The training scripts accepts standard image formats (png, jpeg etc). Instead of defining a class for each file, images corresponding to a class should be placed in subfolders of the data directory. For example:

```
data
 ┣ label1
 ┃ ┣ example1.png
 ┃ ┗ example2.png
 ┣ label2
 ┃ ┣ example1.png
 ┃ ┗ example2.png
```

In both the training and tuning scripts, image manipulation is handled in the same way. 

The directory where the training images are kept, as well as the size the images should be resized to should be specified. 

```python
train_images_dir = './data/train/png/'
image_resize_shape = (28, 28)
```

Different image transformation can then be made. A default has been configured, but can be configured by using the [torchvision transforms][https://pytorch.org/vision/stable/transforms.html] api. For example:

```python
transformer = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(image_resize_shape),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
```

It's advisable to ensure that the transform will keep the image output size consistent with the size specified in `image_resize_shape` as this will be used to configure the CNN model.

The above preperation steps are consistent in both the training (train.py) and tuning (tuning.py) scripts. 

### training.py

The hyperparameters needed for the network training are defined below the image processing.

```python
n_epoch = 15
batch_size = 100
learning_rate = 0.01
```

The optimizer and loss function should also be defined here. Eg..

```python
loss_fn = CrossEntropyLoss
optim = Adam
```

Once trained, the model can be saved to ./models/.

```
model_save_path = './models/fashion_mnst_cnn.bin'
t.save(model.state_dict(), model_save_path)
```

### tuning.py

The hyperparameter tuning script provides a simple utility for exhaustively checking all defined hyperparameter combinations. Only a small amount of hyperparameter tuning was implemented due to the resources available. Hyperparameters can be defined using the tuning dictionary by adding parameters to the corresponding lists below as below.

```python
tuning_grid = {'learning_rate': [0.001, 0.01],
               'n_epoch': [5, 10, 15],
               'batch_size': [100, 500, 1000, 2000]}
```

The proportion of data available to the validation set can be altered by changing the variable *val_size* to a value between 0 and 1.

A list of ditionaries with every parameter combination is generated and looped over, the accuracies of each run are appended to the `accuracies` list. The script prints the best hyperparameter combination on the validation set to the terminal, which can then be used in *train.py* to train and save the model and evaluate on a test set.