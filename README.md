# **hachi-machi**

## Description

**hachi-machi** (_**H**igh-level **A**nd **C**ontrollable **H**uman **I**nterface for **MACH**ine **I**mprovisation_) is an open-source tool for performing artists to easily prototype, train, and run their own improvisational models, with their own data, in their own computers. _hachi-machi_ works as a command-line interface (CLI) written in Python, with PyTorch as its deep learning framework. It's designed to provide the most control to artists in the data-to-model pipeline without any programming involved.

### How it works

To train a model in _hachi-machi_, all you need is a data set to train the model. While _hachi-machi_ provides out-of-the-box training on MIDI files directly, the architecture is, in principle, data agnostic. This means they can be trained on any kind of user-defined sequential data—not musical data, but any type of data that changes over time. In the simplest of cases for custom data models, the data must be provided in the form of a JSON file, as follows:

```json
[
	[<ms_time_0>, <feature_0_0>, ..., <feature0_N>],
	[<ms_time_1>, <feature_1_0>, ..., <feature1_N>],
	[<ms_time_2>, <feature_2_0>, ..., <feature2_N>],
	...
	[<ms_time_M>, <feature_M_0>, ..., <feature_M_N>]
]
```

By default, all features are considered continuous. However, it is also possible to mark certain features as categorical, which is useful for features such as class identifiers. In that case, the features can be specified in the JSON file as such:

```json
{
	"features": {
		"1": { "type": "categorical" }
	},
	"data" [
		[<ms_time_0>, <feature_0_0>, *<feature_0_1>* ...],
		[<ms_time_1>, <feature_1_0>, *<feature_1_1>* ...],
		...
		[<ms_time_N>, <feature_N_0>, *<feature_N_1>* ...]
	]
}
```

This means feature at index `1` (not including time), will be treated as a categorical feature, which improves model training.

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
