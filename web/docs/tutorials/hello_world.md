---
sidebar_position: 0
slug: basics
---

# Basic workflow

This tutorial provides a quick overview of the workflow in **hachi-machi**, by showing how to train and run a basic model for [MIDI](https://en.wikipedia.org/wiki/MIDI) event generation. In this tutoral, you will learn how to:

1. Format MIDI data for training.
2. Train your first model.
3. Generate MIDI-style data with the model (_offline_).
4. Run the model for interactive MIDI generation (_real-time_), via [Open Sound Control](https://en.wikipedia.org/wiki/Open_Sound_Control).

![workflow](@site/static/img/hachi_machi_workflow.svg)

_A visual representation of a workflow in **hachi-machi**._

### Requirements

In addition to [installing](../installation.md) **hachi machi**, you will need the following for this tutorial:

1. A test MIDI file of your choosing.
2. Basic understanding of how to run commands in the terminal.

---

## Formatting our data

To train our models, we need to convert our data to a format **hachi-machi** will understand. In this case, we will use the `format` command to convert our MIDI file to JSON. To start, we run the following command from the directory where our MIDI file is located.

```bash
hxmx format file.mid data.json
```

This will generate a JSON file in the same directory, called `data.json`. This file will look something like this:

```json
{
  "features": { ... },
  "time": [ ... ],
  "data": [ ... ]
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

## Generating sequences

To quickly test our model, we can generate a new sequence of MIDI notes _off-line_ (e.g., not in real-time), using the `gen` command.

```bash
hxmx gen model.pt output.csv
```

This will create a CSV file with our model-generated sequence.

---

## Real-time interaction

To see our model work in real-time, we use the `run` command.

```bash
hxmx run model.pt
```

This will expose the model to receive input and send output messages via [OSC](https://en.wikipedia.org/wiki/Open_Sound_Control). Input messages are received on the `/input` route, and output messages are sent via the `/output` route. To let the model generate data autoregressively, all we need to do is feed the messages from `/output` back to `/input`.

## A word of advice

> 🙌 Machines ≠ humans 🙌

As obvious as this statement is, it's easy to have unrealistic expectations of what our models can and cannot do. We might, for instance, expect this model to have learned human-level understanding of concepts such as harmony and rhythm.

As we'll see in the next tutorials, while there are things we can do to meaningfully improve our models, they will almost always require some degree of "manual" (i.e., rule-based) tweaking on our side. In the case of our MIDI model, this can mean, for instance, applying quantization to the pitch values so that it adheres to some harmonic scheme, or clipping velocity values that ocassionally go outside of the `0-127` range. Naturally, the type of post-processing we need will vary based on factors such as use case, data quality, and training parameters.

Next we'll take a look at how to format our data to improve our models' performance.
