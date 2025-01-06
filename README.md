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

Keynotes:

* Keras offers different ways to build architectures of progressive level of complexity
* One can build models via the Sequential class, via the Functional API, or by subclassing the Model class. 
* In-built `fit()` and `evaluate()` methods are the simplest ones to train and evaluate a model
* Keras callbacks mechanism provides a simple way to monitor train progress of models and automatically take action 
based on the state of the model (fe. to save the best one)
* Full control of fit() is acquired via overriding the `train_step()` method.
* One can write own training loops entirely from scratch (fe. when implementing brand-new training algorithms and
 deigning new training regime)

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

# Timeseries tasks

LSTM can be described as a chain of cells processing the sequence. 
The cells learn to process the sequence sequentially. Each cell learns to adapt its internal state based on the current input 
and the previous state, allowing it to understand the order and context of the sequence. 
This "contextualized" processing enables the prediction of the next step based on its learned sequence model and output at each step.

LSTM layer is a stack of LSTM cells (usually defined by 'units' parameter in libraries like Tensorflow/Pytorch) 
where cells are repeated for every element of input sequence and every stack of cell process current sequence of input.

The power of LSTM is to solve the issues of:

* Vanishing Gradients: During backpropagation (the learning process), the gradients can become extremely small as they 
flow backward through time. This makes it difficult for the network to learn long-range dependencies 
(relationships between distant elements in the sequence). The earlier layers get very little signal for learning.
* Exploding Gradients: Gradients can also become extremely large, leading to unstable learning.

LSTMs address these issues by introducing the concept of "memory" and "gates." Unlike traditional RNNs which use a single hidden state that is overwritten at every step, LSTMs have both a hidden state and a cell state.

* Cell State: This is the long-term memory that can flow from one time step to the next with minimal changes, gated by forget gate.
* Hidden State: This is a short-term memory that is influenced by current input and past cell state. 
  This state influences the output of the current time step

* Gates: These are parameterized NNs that control how information flows into, out of, and within the cell state. 
   The three main types of gates are (marked by sub-index, f.e. W_f):
     1)   **Forget Gate**: Determines what information to remove from the cell state.
     2)   **Input Gate**: Determines what new information to add to the cell state.
     3)   **Output Gate**: Determines what information to output from the cell state into the hidden state.


1) Forget Gate (f_t):

The forget gate decides what information to discard from the cell state (C_t-1).
It takes h_t-1 and x_t as input and passes them through a sigmoid activation function (output between 0 and 1).

The output f_t acts as a filter for C_t-1 where 1 keeps the information and 0 removes the information

Example: In a language model, it might learn to forget the gender of a subject when starting a new sentence

2) Input Gate (i_t and C_t)

i_t: The input gate layer decides which values to update in the cell state

Example: In a language model, it might learn to add a new subject into the cell state for the current sentence

3) Output Gate (o_t):

The output gate controls what information is outputted from the cell state into the hidden state.

Example. Let's assume `[f0, f1, f2, f3]` is the input sequence and the LSTM stack consists of 3 cells.
Initialization: The LSTM's hidden state and cell state are initialized (often to zeros).
Time Step Processing:

For f0: The LSTM cell takes f0 and the initial hidden and cell states. It updates its hidden state and cell state and produces output.
For f1: The LSTM cell takes f1 and the updated hidden and cell state (from f0 processing). It updates its hidden state and cell state and produces output.
And so on... This continues for all the sequence elements (e.g. f2 in our case).

Layer Output:

* If return_sequences=False: the output of the LSTM layer is the last hidden state of the LSTM cell after processing all of input sequences.
* If return_sequences=True: the output of the LSTM layer is all hidden states of the LSTM cell after processing all of input sequence.


Notebook:
[Timeseries analysis](./notebooks/time_series.ipynb)

# NLP tasks, or Text processing



Notebook:
[NLP models](./notebooks/nlp_models.ipynb)

# Transformers

## Notes on embedding spaces


Let's define the original notion/vector as the initial data representation, which might be raw pixels of an image, 
words in a vocabulary, or any other kind of structured data.

Embedding Space is a lower-dimensional space where data points are mapped to, with the goal of having data points 
with similar meaning or relationships placed closer together in this space. This is often learned through training.

The embedding space isn't just some arbitrary mapping; it's designed to highlight the features relevant to the task. 
These features are "expressive" because they capture the important nuances of the data.

Embedding space (or embedding representation) can be viewed as the space of forms or expressive features, which the 
original notion/vector can be factored into. For example, in CNN the space of filters on each layer is effectively the 
representation space, and the original picture can be successfully represented as a set of weighted filters, instead of
grid of pixel values.

The example of CNNs provides an excellent illustration:

Consider the original picture. A grid of pixel values is essentially a very high-dimensional, low-level representation. 
It lacks abstraction and semantic meaning.

CNN filters are designed to detect specific patterns (edges, textures, shapes, etc.) in the input image. 
These filters are the "forms" or "expressive features" the statement is talking about.
Each layer in a CNN effectively creates its own embedding space. The activations from each filter at a particular layer 
can be seen as a feature vector representing the image at that level of abstraction.

The CNN learns weights for these filters through backpropagation. The activations of the filters, scaled by those weights, 
represent a higher-level and more "meaningful" representation of the image than the original pixels. 
Instead of grid of pixel values, the image is now successfully represented as weighted combination of filters which 
detects the underlying feature of the image.

Why this is important

* Dimensionality Reduction: Embeddings often have a much lower dimension than the original data, allowing us to handle 
  large datasets more efficiently and avoid the "curse of dimensionality."
* Semantic Meaning: Embedding spaces are designed to capture the semantic meaning of the data.
  Similar items are grouped together, making it easier for models to learn and make inferences.
* Feature Extraction: The process of learning an embedding essentially learns the relevant features for the given task. 
  This eliminates the need for manual feature engineering.
* Transfer Learning: Pre-trained embeddings, learned on large datasets, can be used for new tasks, 
  leveraging the knowledge about feature spaces that have already been acquired.
* Generative Capabilities: By working with the embedding space, we can manipulate and generate new instances of the data. 
  For example, with image embeddings, one can interpolate between different image embeddings to generate similar images.


## Transformer as a Transformation Between Embedding Spaces:

Transformer performs a transformation of embedding spaces using the following steps:

1) Input Embedding:
    The encoder begins by mapping input tokens into an initial embedding space. Let's call this space _E_input_.
2) Encoder Transformation:
    The encoder then applies a series of self-attention layers and feedforward networks. These layers are all 
    essentially linear transformations (through the feedforward networks) combined with weighted aggregates via the attention. 
    This process transforms the input embeddings from _E_input_ into a higher-level representation, let's call it _E_encoder_. 
    _E_encoder_ is a representation of the input in the form of context-aware embeddings.
3) Decoder Transformation:
    The decoder takes the E_encoder representation from the encoder and the target embeddings as inputs. 
    The decoder then uses attention mechanisms, and feedforward networks to transform and translate this information 
    to a new embedding space. Let's call this _E_output_. The E_output space represents the target sequence representation.
4) Final Output Transformation:
    Finally, a linear layer often projects this space _E_output_ to the desired target space (e.g. probability scores for words in a vocabulary).

Constraints:

* Transformation Parameters: The parameters (weights and biases) of the Transformer layers and attention heads are 
the "transformation parameters." These parameters are learned during training through backpropagation, guided by the loss function.
* Loss Function: The loss function defines how "good" the transformation is. 
  It quantifies the difference between the model's output and the true target. 
  For example, cross-entropy loss is often used for classification (like next-word prediction) or sequence-to-sequence tasks. 
  The loss acts as a constraint, forcing the transformation to map the input space to the correct output space according to your task.

Key Insights:

Abstract Representation: The Transformer's power comes from its ability to learn meaningful intermediate representations 
(E_encoder and E_output). These are not simply direct mapping between words in different spaces; 
instead, they are abstract representations that encode syntactic and semantic relationships.

Learned Transformations: The magic of neural networks is that these transformation operations are learned from data and 
guided by a loss function. That's why transformers can perform translation, text generation, and other sophisticated tasks.

Generalization: The learned transformations can often generalize to unseen data, allowing the model to perform well 
on new examples from similar distributions.


Notebook:
[Transformers](./notebooks/transformers.ipynb)

# Transformers for translation

Notebook:
TBA

# References
(Note, the key papers are duplicated in the folder [papers](./papers) if license allows)
