# Operations

Some commands, such as `train` or `augment`, support passing a series of optional operations via the `--operations` option.

Operations provide a flexible way to design data augmentation pipelines, by composing a sequence of data operations to be applied to the data, specified as python-like functions calls.
For instance, consider a configuration file that specifies the following data augmentation operations:

import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";

<Tabs groupId="config-files">
  <TabItem value="yaml" label="yaml">
  ```yaml title="train_config.yaml"
  cmd: train
  input: data.json
  operations:
    - sub(0, value=mean)
    - div(0, value=std)
    - randmul(0, a=0.9, b=1.1, dist=uniform)
    - mul(0, value=std)
    - add(0, value=mean)
  ```
  </TabItem>
  <TabItem value="toml" label="toml" default>
    ```toml title="config.toml"
    cmd = "train"
    input = "data.json"
    operations = [
      "sub(0, value=mean)",
      "div(0, value=std)",
      "randmul(0, a=0.9, b=1.1, dist=uniform)",
      "mul(0, value=std)",
      "add(0, value=mean)"
    ]
    ```
  </TabItem>
</Tabs>

:::info
Note that text-like values such as `mean` or `normal` are not in quotes.
:::

In this example, the following series of operations will be applied to feature at dim `0` in each sequence in the training batch:

1. Subtract the mean
2. Divide by its standard deviation
3. Randomly multiply it by some uniformly distributed random value between `0.9` and `1.1`.
4. Multiply by standard deviation.
5. Add mean to bring feature back to initial scale.

:::warning
For any augmentation pipeline, always ensure your data always remains in a realistic range.
:::

The following are all currently available operations for data augmentation during training.
