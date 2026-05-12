---
sidebar_position: 0
---

# 1. Basic workflow

This tutorial provides a quick-and-dirty overview of the workflow in **hachi-machi**, by training and running a custom model for MIDI sequence generation. This workflow can be summarized as follows:

![workflow](@site/static/img/hachi_machi_workflow.svg)

For this tutorial, you will need:

1. A test MIDI file of your choosing.
2. Basic understanding of how to run commands in the terminal.

---

## Creating a dataset

To train our models, we need to convert our data to a format **hachi-machi** will understand. In this case, we will use the `format` command to convert our MIDI file to JSON. To start, we run the following command from the directory where our MIDI file is located.

```bash
hxmx format file.mid data.json
```

This will generate a JSON file in the same directory, called `data.json`. This file will look something like this:

```json
{
  "features": {...},
  "time": [...],
  "data": [...]
}
```

We'll discuss later how to better understand each of these blocks. For now, let's move to the next step: **training our first model**.

---

## Training the model

Once the data is ready, we tell **hachi-machi** to create a new model trained on our JSON-formatted MIDI data, using the `train` command, like so:

```bash
hxmx train data.json model.pt
```

This will quick-off the training loop. Depending on the length of the MIDI file we're using, and our computer's capabilities, this might take from one to several minutes (or even hours if you're unlucky!). We should see our new model in the same directory, with the name `model.pt`.

---

## Sequence generation

To quickly test our model, we can generate a new sequence of MIDI notes _off-line_ (e.g., not in real-time), using the `gen` command.

```bash
hxmx gen model.pt output.csv
```

This will create a CSV file with our model-generated sequence.

---

## Real-time generation

To see our model work in real-time, we use the `run` command.

```bash
hxmx run model.pt
```

This will expose the model to receive input and send output messages via [OSC](https://en.wikipedia.org/wiki/Open_Sound_Control). Input messages are received on the `/input` route, and output messages are sent via the `/output` route. To let the model generate data autoregressively, all we need to do is feed the messages from `/output` back to `/input`.
