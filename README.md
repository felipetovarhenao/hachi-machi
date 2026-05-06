# **hachi-machi**

## Description

**hachi-machi** (_**H**igh-level **A**nd **C**ontrollable **H**uman **I**nterface for **MACH**ine **I**mprovisation_) is an open-source tool for performing artists to easily prototype, train, and run their own improvisational models, with their own data, in their own computers. _hachi-machi_ works as a command-line interface (CLI) written in Python, with PyTorch as its deep learning framework. It's designed to provide the most control to artists in the data-to-model pipeline without any programming involved.

## How it works

To train a model in _hachi-machi_, all you need is a set of sequential data to train the model. While _hachi-machi_ works out-of-the-box for training models of MIDI data, the model's architecture is, in principle, data agnostic. This means they can be trained on any kind of user-defined sequential data—not only musical data, but any other type of data that in sequential in nature.

### Custom data models

#### Non-temporal data

The most trivial type of model is a _non-temporal_ model, meaning a model that is trained on sequential data, where there is no need for the model to have a sense of time, but only of order. Examples could include, for instance, predicting the next chord in a chord progression. In this case, the data should be structured as follows:

```json
{
	"data": [
		[<feature[0,0]>, ..., <feature[0,N]>],
		[<feature[1,0]>, ..., <feature[1_N]>],
		[<feature[2,0]>, ..., <feature[2,N]>],
		...
		[<feature[M,0]>, ..., <feature[M,N]>]
	]
}
```

#### Temporal data

The more relevant case is models where the timing of an event is important, the content of the prediction is as important as when the prediction is supposed to happen. A clear example would be to predict MIDI notes, which not only requires predicting information such as pitch, velocity, but also when that note happens. In such case, the data must be structured as follows:

```json
{
	"time": [
		<ms_time[0],
		<ms_time[1]>,
		...,
		<ms_time[N]>
	]
	,
	"data": [
		[<feature[0,0]>, ..., <feature[0,N]>],
		[<feature[1,0]>, ..., <feature[1,N]>],
		[<feature[2,0]>, ..., <feature[2,N]>],
		...
		[<feature[M,0]>, ..., <feature[M,N]>]
	]
}
```

Where time denotes the onset, in milliseconds, at which each event in `"data"` happens. Not that the length of `"time"`, must match the number of events in `"data"`.

### Feature types

By default, all event features are considered continuous—i.e., they can take any value within a range. An example of continuous features include pitch, amplitude, duration. However, it is also possible to mark certain features as discrete or _categorical_, which is useful for features such as class identifiers—e.g., instrument ID, ON/OFF flags, etc.. In that case, the features can be marked as such in the JSON file as such:

```json
{
	"features": {
		"1": { "type": "categorical" }
	},
	"data": [
		[<feature[0,0]>, *<feature[0,1]>*, ..., <feature[0,N]>],
		[<feature[1,0]>, *<feature[1,1]>*, ..., <feature[1,N]>],
		[<feature[2,0]>, *<feature[2,1]>*, ..., <feature[2,N]>],
		...
		[<feature[M,0]>, *<feature[M,1]>*, ..., <feature[M,N]>]
	]
}
```

This means feature at index `1` (highlighted with asterisk), will be treated as a categorical feature, which improves model training.

---

## Setup

1. Install [conda](https://anaconda.org/):
2. In the terminal, run:

```sh
conda create -n machi python=3.10
conda activate machi
conda install pytorch click
python -m pip install python-osc mido toml
pip install -e .
```

## Before each session

```sh
conda activate machi
```

---

## Usage

### Training

```zsh
machi train <./input.mid> <./output.pt>
```

### Generation (_offline_)

```zsh
machi generate <./model.pt> <output.mid>
```

### Streaming (_real-time_)

```zsh
machi run <./model.pt>
```

### MIDI rendering

```zsh
machi render <./input.mid> <./output.mid>
```
