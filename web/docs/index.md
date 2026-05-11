---
sidebar_position: 0
---

# Getting started

## Description

**hachi-machi** (_**H**igh-level **A**nd **C**ontrollable **H**uman **I**nterface for **MACH**ine **I**mprovisation_) is an open-source tool for performing artists to easily prototype, train, and run their own improvisational models, with their own data, in their own computers. _hachi-machi_ works as a command-line interface (CLI) written in [Python](<(https://www.python.org/)>), with [PyTorch](https://pytorch.org/) as the back-end for deep learning. It's designed to provide the most control to artists in the data-to-model pipeline without any programming involved.

### How it works

To train a model in _hachi-machi_, all you need is a set of sequential data to train the model. While _hachi-machi_ works out-of-the-box for training models of MIDI data, the model's architecture is, in principle, data agnostic. This means they can be trained on any kind of user-defined sequential data—not only musical data, but any other type of data that in sequential in nature.

### Custom data models

Training a custom model requires a JSON file describing your dataset. The exact structure depends on whether timing information is relevant to your use case.

#### Non-temporal data

Use this format when event order matters but timing does not—for example, predicting the next chord in a chord progression.

```json
{
    "data": [
        [<feature_0>, ..., <feature_N>],
        [<feature_0>, ..., <feature_N>],
        ...
    ]
}
```

`"data"` is a list of events, where each event is a list of `N` feature values. Features are ordered consistently across all events.

#### Temporal data

Use this format when the timing of each event is part of what the model should learn—for example, predicting MIDI notes requires knowing not just pitch and velocity, but when each note occurs. This will determine how temporal vs non-temporal models behave during _streaming_ mode (via the `run` command): the prediction will be emitted immediately for non-temporal models, while the prediction is scheduled to be emitted at some time in the future based on what the model learned.

```json
{
    "time": [
        <time_0>,
        <time_1>,
        ...
    ],
    "data": [
        [<feature_0>, ..., <feature_N>],
        [<feature_0>, ..., <feature_N>],
        ...
    ]
}
```

`"time"` is a list of onset times in seconds, one per event. Its length must match the number of events in `"data"`.

#### Feature types

By default, all features are treated as **continuous**—that is, they can take any numeric value within a range (e.g., pitch, amplitude, duration).

Features can also be declared **categorical**, which is appropriate for discrete identifiers such as instrument IDs or ON/OFF flags. Doing so improves model training by changing how those features are encoded internally.

To mark a feature as categorical, add a `"features"` entry to the JSON file, keyed by the feature's zero-based index:

```json
{
    "features": {
        "1": { "categorical": true }
    },
    "data": [
        [<feature_0>, <feature_1>, ..., <feature_N>],
        ...
    ]
}
```

In this example, the second feature in each event (index `1`) will be treated as categorical. Any feature index not listed in `"features"` is assumed continuous.

## Installation

1. Download [Python](https://www.python.org/)
2. In the terminal, run:

```zsh
pip install hachi-machi
```

3. To check that's successfully installed, run:

```zsh
hxmx --version
```
