---
sidebar_position: 3
slug: config-files
---

import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";

# Configuration files

**hachi-machi** supports executing commands through configuration files instead of passing them directly in the terminal, as either [YAML](https://www.datacamp.com/blog/what-is-yaml) or [TOML](https://toml.io/) files, via the `exec` command. This is particularly useful for model training, which has numerous parameters that can quickly get out of hand and hard to tinker with.

All configuration files require specifying the command to execute via the `cmd` key along with any required arguments for that command. Below is an example of a basic configuration file for the `gen` command:

<Tabs groupId="config-files">
  <TabItem value="yaml" label="yaml">
    ```yaml title="config.yaml"
    cmd: gen
    input: ./mymodel.pt
    output: seq.txt
    tokens: 300
    ```
  </TabItem>
  <TabItem value="toml" label="toml" default>
    ```toml title="config.toml"
    cmd = "gen"
    input = "./mymodel.pt"
    output = "seq.txt"
    tokens = 300
    ```
  </TabItem>
</Tabs>

Then, we can run:

<Tabs groupId="config-files">
  <TabItem value="yaml" label="bash (yaml)">
  	```bash
    hxmx exec config.yaml
    ```
  </TabItem>
  <TabItem value="toml" label="bash (toml)" default>
    ```bash
    hxmx exec config.toml
    ```
  </TabItem>
</Tabs>
