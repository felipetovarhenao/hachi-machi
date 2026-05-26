# Operations

Some commands, such as `train` or `fork`, support passing a series of optional operations via the `--operations` option.

Operations provide a flexible way to design data augmentation pipelines, by composing a sequence of primitive data operations to be applied to the data, specified as **case-insensitive**, python-like functions calls.

![operations](@site/static/img/hachi_machi_ops.svg)

For instance, consider a configuration file that specifies the following data augmentation operations:

import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";

<Tabs groupId="config-files">
  <TabItem value="yaml" label="yaml">
  ```yaml title="train_config.yaml"
  cmd: train
  input: data.json
  operations:
    - sub(value=mean, axis=feature)
    - div(value=std, axis=feature)
    - mulrand(range=(-0.1, 0.1), log=true)
    - mul(value=std, axis=feature)
    - add(value=mean, axis=feature)
  ```
  </TabItem>
  <TabItem value="toml" label="toml" default>
    ```toml title="config.toml"
    cmd = "train"
    input = "data.json"
    operations = [
      "sub(value=mean, axis=feature)",
      "div(value=std, axis=feature)",
      "mulrand(range=(-0.1, 0.1), log=true)",
      "mul(value=std, axis=feature)",
      "add(value=mean, axis=feature)",
    ]
    ```
  </TabItem>
</Tabs>

:::info
Note that text-like values such as `mean` or `std` are not in quotes.
:::

In this example, the following series of operations will be applied to the input data:

1. Subtract the feature-wise mean
2. Divide by its feature-wise standard deviation
3. Multiply (i.e., scale) by some normally distributed random value between `2 ** -0.1` and `2 ** 0.1`.
4. Multiply by its feature-wise standard deviation.
5. Add feature-wise mean to bring feature back to initial scale.

:::caution
For any augmentation pipeline, make ensure your data always remains in a realistic range. Poorly designed data augmentation pipelines might result in worse training outcomes.
:::
