---
sidebar_position: 4
slug: data-augmentation
---

import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";

# Data augmentation

One of the main challenges for artists to experiment with deep-learning models using their own data, is the quantity requirements relative to what is typically expected in industry standards.

One way to get around this is through [data augmentation](https://www.ibm.com/think/topics/data-augmentation), which is to say, applying random variations to our data during training, so as to artificially increment its size. However, the particular approach we take will necessarily vary, not just on the kind of data we're using, but also on _what_ we want the model to learn about the data.

To this end, **hachi machi** provides a series of [operations](/docs/commands/operations) that can be specified during training, and applied in series to each sequence example in a training batch.

Each operation is specified as a list of python-like function calls, via the `--operations` parameter, which is available in the `train` and `fork` commands.

---

## Augmenting MIDI data

To continue with the MIDI data example, let's remember how our data is formatted:

<Tabs groupId="data">
    <TabItem value="json" label="json">
    ```json
    {
        "time": [0, 0.25, 0.5, 0.75, 1.0],
        "data": [
            [60, 80, 0.25, 0],
            [64, 75, 0.25, 1],
            [67, 85, 0.25, 0],
            [64, 70, 0.25, 1],
            [60, 90, 0.5, 0]
        ]
    }
    ```
    </TabItem>
        <TabItem value="csv" label="csv">
    ```csv
    time,f0,f1,f2,f3
    0.0,60,80,0.25,0
    0.25,64,75,0.25,1
    0.5,67,85,0.25,0
    0.75,64,70,0.25,1
    1.0,60,90,0.5,0
    ```
    </TabItem>
</Tabs>

Where each event, in addition to having a time position, consists of the following features: MIDI pitch, velocity, duration (_seconds_), and channel. Now consider the following series of operations and the likely effects each of them will have on the model.

<Tabs groupId="config">
    <TabItem value="yaml" label="yaml">
    ```yaml
    cmd: train
    input: midi.csv
    output: model.pt
    operations:
    # random MIDI pitch transposition
    - addrand(range=(-6, 6), dims=0)
    # random velocity scaling
    - mulrand(range=(0.5, 1), dims=1)
    # random time stretching, applied to time and duration
    - mulrand(dims=(t, 3), range=(-0.5, 0.5), log=true)
    ```
    </TabItem>
     <TabItem value="toml" label="toml">
    ```toml
    cmd = 'train'
    input = 'midi.csv'
    output = 'model.pt'
    operations = [
        # random MIDI pitch transposition
        'addrand(range=(-6, 6), dims=0)',
        # random velocity scaling
        'mulrand(range=(0.5, 1), dims=1)',
        # random time stretching, applied to time and duration
        'mulrand(dims=(t, 3), range=(-0.5, 0.5), log=true)',
    ]
    ```
    </TabItem>
</Tabs>

---

### Pitch transposition

```rust
addrand(range=(-6, 6), dims=0)
```

The first feature (dim `0`) in each sequence contains MIDI pitch values. This operation randomly shifts each pitch by an integer value drawn uniformly from the range `[-6, 6]`, effectively transposing the passage by up to a tritone up or down. Ostensibly, this encourages the model to learn melodic contours and intervals more robustly, making it less susceptible to pitch transpositions.

---

### Velocity scaling

```rust
mulrand(range=(0.5, 1), dims=1)
```

The second feature (dim `1`) contains MIDI velocity values, which correspond to note loudness. This operation randomly scales each velocity by a value drawn from the range `[0.5, 1]` applying variable dynamic compression. This encourages the model to learn patterns that are robust to differences in overall loudness.

---

### Time stretching

```rust
mulrand(dims=(t, 3), range=(-0.5, 0.5), log=true)
```

This operation targets both the time axis (`t`) and the duration feature (dim `3`) simultaneously, scaling them by the same random factor drawn from `[-0.5, 0.5]` in log space. Applying the same factor to both ensures that the relative timing between notes remains internally consistent, while the overall tempo varies. This encourages the model to be more responsive to tempo changes, though it also means its output will be perceived as less _rhythmic_.

:::info
By _log space_, we mean that the random number is interpreted in a _log base 2_ space, meaning in the `2 ** -0.5` to `2 ** 0.5` range.
:::

:::tip
Even though temporal datasets include the absolute time position of each step in the sequence, **hachi machi** replaces it with the time difference between the current and previous step. As such, operations of the `t` dimension are not applied not to absolute time position but to these differences.
:::
