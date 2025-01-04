# Tensorflow-grimoire

This repository explores the fundamentals of Deep Learning on examples using TensorFlow framework

# Fundamentals of ML, Tensorflow and numpy

## Parameters in Dense Layers

Dense (or fully connected) layers in neural networks are the primary contributors to the total parameter count. 
Here's how the number of parameters is calculated for a dense layer:

* Weights: Each input to a neuron in the dense layer has a weight associated with it.
* Biases: Each neuron also has a bias (a constant value) added to its weighted input.

The formula for calculating the parameters in a single dense layer is:

```
num_params = (input_size * output_size) + output_size
``` 

For example, for the model:

```
model = keras.Sequential()
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
model.build(input_shape=(None, 3))
```

1st layer - input 1-tensor (3,) -> output 1-tensor (32,0)  will require a transformation matrix with 32 * 3 weighs + 
bias vector with 32 neurons, total number of params = 32 * 3 + 32 = 128

Generalization: look at input and output shape and calculate the dimensionality and volume of transformation matrix.

# Advanced topics in Tensorflow

## Sequential Models, Functional API and Subclassing

* The Sequential model essentially is a Python list(a simple stacks of layers).
* The Functional API treat the model as a graph-like architectures, is a balance between usability and flexibility
* Model subclassing, a low-level option allowing to implement everything from scratch

In Functional API connections are represented by input and output tensors (normally of rank 1), and the nodes in graph are transformations 
(Keras layers).

The mixing of input/output could be of two types:

1) concatenation of tensors/features (i.e. vectors become larger)
2) vector sum
3) adding as new dimensions 

# Custom metrics

Metrics are key to measuring the performance of model, f.e. to measuring the difference between its performance on the 
training data and test data. Commonly used metrics for classification and regression are already part of the built-in 
`keras.metrics` module.

A custom Keras metric is a subclass of the `keras.metrics.Metric` class. Like layers, a metric has an internal state 
stored in TensorFlow variables, but those variables are not updated via backpropagation (need to implement an
`update_state()` method)


Notebook:
[Sequential Models, Functional API and Subclassing](./notebooks/advanced-tf-keras.ipynb)


# Deep Learning in Computer Vision tasks

## Architecture

CNN networks are built on the basis of hierarchy of features, constructed on each level via
```
 layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
```
where

`filters=32`: Number of Filters (Output Channels) - specifies how many different filters are applied to the input. 
In this case, we have 32 filters. Each filter is a unique feature detector, f.e. some filters might be good at detecting 
horizontal edges, others at detecting vertical edges, and others might look for textures, corners, etc.
For each filter, the convolution operation creates a new output "image". 
Because we have 32 filters, our layer will output 32 new images (aka "feature maps" - the output 
will have the shape (?, ?, 32)

`kernel_size=3` : Size of the Filter (Kernel) - defines the size of each filter. kernel_size=3 means each filter is a 3x3 grid.
The filter has params and is slid over the input image. At each location, a dot product is calculated between the 3x3 
input "window" of the image and the 3x3 filter's numbers and becomes one pixel value in the output feature map.

## How the Filters are Generated

Filters have to be the _independent_ filters. 
Each of these 32 filters will learn a different set of weights, resulting in a distinct feature map.

Initialization: When layers.Conv2D is created, TensorFlow (or the underlying framework) will:
1) Create 32 separate filters.
2) Each filter will be a 3x3 matrix.
3) The initial values within each 3x3 matrix are typically randomly initialized. This randomness is _crucial_ because it 
   allows different filters to learn different features (via Xavier/Glorot or He initialization strategies)

Notebook:
[Deep Learning in Computer Vision tasks](./notebooks/dl_in_computer_vision_tasks.ipynb)

## Interpretability of CNN

Interpretability means the ability to track back the causes and weights which resulted in specific decision made by NN.

In CNNs, feature maps are the output of each convolutional layers (type `keras.src.layers.convolutional.conv2d.Conv2D`). 
They represent the learned features of the input image at different levels of abstraction. 
The deeper into the network, the more abstract the features become.

Via extracting these specific layers one can generate Class Activation Maps (CAMs). CAMs are visualizations 
that highlight which parts of an image are the most important ones for classification.

For example, in case of Xception NN, `block14_sepconv2_act` is the activation output after the 2nd separable convolution 
in the 14th block. This layer typically has the smallest spatial resolution but the richest semantic content right before 
the classification.

Note: choosing the last convolutional layer is _crucial_ for CAM generation. 
The feature maps at this stage of the network contain a spatial encoding of features relevant for the classification task 
but also are still spatially coherent and can be scaled back onto the input image (i.e. blurred, but points to the most
important region on the picture)

To create a CAM, the new model has to take the same input as the original model, 
but the output will be a feature map produced by the last_conv_layer instead of the classification probabilities.

We need to compute the gradients of the predicted class probability (top_class_channel) with respect to the activations 
of the watched convolutional layer (last_conv_layer_output). 
This gradient tells us how much each spatial location in the convolutional layer's output contributed to the 
model's prediction for the chosen class.

Notebook:
[Interpretability of CNN](./notebooks/interpretability_of_cnn.ipynb)

# 



