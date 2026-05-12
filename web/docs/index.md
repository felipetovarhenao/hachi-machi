---
sidebar_position: -2
---

# Getting started

## Description

**hachi-machi** (_**H**igh-level **A**nd **C**ontrollable **H**uman **I**nterface for **MACH**ine **I**mprovisation_) is an open-source tool for performing artists to easily prototype, train, and run their own improvisational models, with their own data, in their own computers. _hachi-machi_ works as a command-line interface (CLI) written in [Python](<(https://www.python.org/)>), with [PyTorch](https://pytorch.org/) as the back-end for deep learning. It's designed to provide the most control to artists and technologists in the data-to-model pipeline while minimizing the amount of expertise required.

### How it works

![training](@site/static/img/hachi_machi_training.svg)

To train a model with **hachi-machi**, all you need is a set of sequential data to train the model. **hachi-machi**'s model architecture is, in principle, data agnostic. This means they can be trained on any kind of user-defined sequential data—such as musical data, and any other type of data where the order of events is semantically meaningful.
