---
sidebar_position: -1
---

# Installation

1. Download, and install [Python](https://www.python.org/) (_v3.10 or higher_)

:::note[GPU support for non-Apple silicon users]
**hachi machi** installs a CPU-only version of PyTorch by default. Since training models requires GPU acceleration, you will need to manually install the appropriate PyTorch build _before_ installing **hachi machi**. See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to find the right command for your system.
:::

2. In the terminal, run:

```bash
pip install hachi-machi --pre
```

3. Then run:

```bash
hxmx --version
```

If you see a banner in the console, it's been sucessfully installed.
