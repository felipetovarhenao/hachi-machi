# `hachi.machi`: High-level and Controllable Human Interface for Machine Improvisation

**Hachi Machi** (_**H**igh-level **A**nd **C**ontrollable **H**uman **I**nterface for **MACH**ine **I**mprovisation_) is a tool for training and running custom improvisational music models.

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
