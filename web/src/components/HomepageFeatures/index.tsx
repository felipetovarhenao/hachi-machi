import type { ReactNode } from "react";
import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<"svg">>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: "Made for artists",
    Svg: require("@site/static/img/undraw_making-art.svg").default,
    description: (
      <>
        <b>hachi machi</b> is primarily designed for artistic applications. Easily train, prototype, and experiment with steerable models with small,
        bespoke datasets.
      </>
    ),
  },
  {
    title: "Train and run locally",
    Svg: require("@site/static/img/undraw_data-processing.svg").default,
    description: (
      <>
        <b>hachi machi</b> runs entirely on your own machine—no API keys, cloud services, or subscriptions. Your data and models stay in your
        computer.
      </>
    ),
  },
  {
    title: "Data agnostic",
    Svg: require("@site/static/img/undraw_ai-slop.svg").default,
    description: (
      <>
        <b>hachi machi</b> models can be trained on any data that can be represented as a sequence of events—e.g., music, sensor, motion data, etc.
        You decide its application.
      </>
    ),
  },
  {
    title: "Minimal programming",
    Svg: require("@site/static/img/undraw_programming.svg").default,
    description: (
      <>
        The entire process—from preparing data to training and running a model—is handled through a friendly{" "}
        <a target="_blank" href="https://youtu.be/w9u0d4C95Zs">
          command-line interface
        </a>
        .
      </>
    ),
  },
];

function Feature({ title, Svg, description }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.row}>
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
