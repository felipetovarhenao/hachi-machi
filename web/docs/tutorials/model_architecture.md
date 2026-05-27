---
sidebar_position: 6
slug: the-model
---

# Inside the model

**hachi machi** is based on a deep learning architecture described by Alex Graves his 2014 paper, _Generating Sequences With Recurrent Neural Networks_, which he uses for generating hand-writing sequences. At its core, it consists of two types of neural networks connected to each other: a _Long Short Term Memory_ (LSTM) network, and _Mixture Density Network_ (MDN).

## LSTM

A **Long Short Term Memory** is a type of recurrent neural network, or RNN, which is what allows the model to learn from sequential data, and consider not just the current input but also learn a representation of the _history_ of past inputs, as a way to predict what the next step should be. However, since LSTMs produce a single deterministic output for any given input—rather than a range of possibilities—they have limited use in generative contexts where variation and unpredictability are desirable.

## MDN

**Mixture Density Networks** are models that learn how to map some input to a density function of a desired output in the form of a Gaussian mixture—a set of multiple normally distributed functions. In other words, instead of predicting the output directly, it learns to predict the parameters for some combination of multiple normal distributions, such that the probability of the target output is high.

One of the parameters we get to specify in our models is the number of Gaussians the model can learn. In the context of our **LSTM-MDN** architecture, a somewhat simplistic way to think about it is that it determines how many possible paths can the model take given some input to make the next prediction. Intuitively, this means the number of possible sequences the model can generate is substantially higher with each additional mixture, making it well-suited for generative tasks.

As such, here three of the most impactful parameters we can specify during training via the `train` command, that directly affect the number of parameters or weights:

- `mixtures`: The number of Gaussian mixtures in the MDN.
- `hidden size`: The number of dimensions the LSTM can learn to represent the history of previous events.
- `layers`: The number of sequential layers for the LSTM. More layers allow the model to learn sequences hierarchically, but it may also makes the learning more difficult and even lead to worse results.
