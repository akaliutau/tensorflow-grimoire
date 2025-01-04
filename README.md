# Tensorflow-grimoire

This repository explores the fundamentals of Deep Learning on examples using TensorFlow framework

# Fundamentals of Tensorflow and numpy

Parameters in Dense Layers

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

Notebook:
[Deep Learning in Computer Vision tasks](./notebooks/dl_in_computer_vision_tasks.ipynb)

## 



