---
sidebar_position: 1
---

# The golden rule

> 🙌 Machines ≠ humans 🙌

As obvious as this statement is, it's easy to have unrealistic expectations of what our models can and cannot do. More precisely, to get our models to behave as expected, they will almost always require some tweaking on the output side, in some way or another.

In the case of our MIDI model, this can mean, for instance, applying quantization to the pitch feature so that it adheres to some harmonic scheme; or clipping velocity values that ocassionally go outside of the `0-127` range. Of course, the type of post-processing we apply will vary based on the use case.
