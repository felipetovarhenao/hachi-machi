---
sidebar_position: -2
---

import VideoCard from "@site/src/components/VideoCard";

# Getting started

## Description

**hachi machi** (_**H**igh-level **A**nd **C**ontrollable **H**uman **I**nterface for **MACH**ine **I**mprovisation_) is an open-source tool for performing artists to quickly and easily prototype, train, and run their own improvisational models, with their own data, in their own computers. Your data never leaves your computer, and your models are trained from scratch exclusively on the data you provide.

### How it works

**hachi machi** works as a command-line interface (CLI) written in [Python](https://www.python.org/), with [PyTorch](https://pytorch.org/) as the back-end for deep learning—however, no programming is needed to use it. It's designed to provide the most control to artists and technologists in the data-to-model pipeline while minimizing the amount of expertise required.

To train a model with **hachi machi**, all you need is:

1. A computer with a decent GPU. For macOS users, this means any [M-series/Apple Silicon](https://support.apple.com/en-us/116943) computer. For Windows or Linux users, this typically means access to an [NVIDIA](https://www.nvidia.com/) or [AMD](https://www.amd.com/en/products/graphics/desktops/radeon.html) GPU. Alternatively, you may train your model in the cloud (e.g., [Google Colab](https://colab.research.google.com/)), but your data is not guaranteed to remain private.
2. a set of sequential data—that is, data that can be represented as a series of steps, each step consisting of set of numerical features or values.

The role of a **hachi machi** model is to learn how to predict the next event, like so:

![training](@site/static/img/hachi_machi_training.svg)

**hachi machi**'s model architecture is, in principle, [data agnostic](<https://en.wikipedia.org/wiki/Agnostic_(data)>). This means models can be trained on any kind of sequential data—such as musical data, sensor data, or any other type of data where the order of events is semantically meaningful.

<VideoCard videoId="R9vMuaHQfB0" />
