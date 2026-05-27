---
sidebar_position: 4
slug: recording
---

# Recording data

More often than not, we will want to train our models on bespoke data, instead of standardized and readily available formats such as MIDI—e.g., motion or sensor data. To this end, **hachi machi** supports recording sequential data via the `rec` command, which makes it easier to generate custom sequential datasets, in either CSV, or JSON format, ready for model training.

To do so, we must specify, at a minimum, the number of features incoming OSC event are expected to contain. For instance, the following command starts a recording session for a temporal dataset with 4 features:

```
hxmx rec 4 mydata.csv --temporal
```

Once running, we can start sending OSC events to the `/input` route, and send `/stop` to stop recording and write the sequence as a CSV file named `mydata`.

Beyond this, we can provide optional parameters, such as which features should be marked as masked or categorical.
