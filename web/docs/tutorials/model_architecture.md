---
sidebar_position: 6
slug: the-model
---

# Inside the model

**hachi machi** is based on a deep learning architecture described by Alex Graves in his 2013 paper, [_Generating Sequences With Recurrent Neural Networks_](https://arxiv.org/pdf/1308.0850v5), which he uses to generate text and hand-writing sequences. At its core, it consists of two types of neural networks connected to each other: a _Long Short-Term Memory_ (LSTM) network, and _Mixture Density Network_ (MDN).

## Long Short-Term Memory

A **Long Short-Term Memory** (LSTM) is a type of [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network), or RNN, which allows the model to learn from sequential data, and consider not just the current input but also learn a representation of the _history_ of past inputs, as a way to predict what the next step should be. However, since LSTMs produce a point estimate rather than a probability distribution over possible outputs, they have limited use in generative contexts where variation and unpredictability are desirable.

## Mixture Density Network

**Mixture Density Networks** are models that learn to output the parameters of a Gaussian mixture model—a weighted combination of normal distributions—such that the probability of the target output is high

![Gaussian mixture](@site/static/img/gaussian_mixture.svg)

One of the parameters we get to specify in our models is the number of Gaussians the model can learn. In the context of our **LSTM-MDN** architecture, a somewhat simplistic way to think about it is that it determines how many possible paths the model can take given some input to make the next prediction. Intuitively, more mixtures allow the model to approximate more complex distributions, capturing a wider variety of patterns in the training data.

As such, the main parameters we can provide during training, via the `train` command, that directly affect the number of parameters or weights are:

- `--mixtures`: The number of Gaussian mixtures in the MDN—i.e., how many _fuzzy_ decision branches should the model learn.
- `--hidden-size`: The dimensionality of the LSTM's hidden state vector at each timestep—i.e., the capacity of the model's learned representation at each step in the sequence.
- `--layers`: The number of sequential layers for the LSTM. More layers allow the model to learn sequences hierarchically, though it also increase the risk of overfitting and training instability, with diminishing returns beyond a certain depth. In many cases, a single layer is enough.
