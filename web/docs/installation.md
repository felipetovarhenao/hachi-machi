---
sidebar_position: -1
---

# Installation

1. Download, and install [Python](https://www.python.org/) (_v3.10 or higher_)

:::note[GPU support]
**hachi machi** installs a CPU-only version of PyTorch by default. If you want GPU acceleration, install the appropriate PyTorch build _before_ running the command below. See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to find the right command for your system.

Apple Silicon (MPS) requires no additional steps.
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
