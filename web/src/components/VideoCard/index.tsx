import React, { useEffect, useState } from "react";
import styles from "./styles.module.css";
import clsx from "clsx";

interface VideoCardProps {
  videoId: string;
}

interface OEmbedResponse {
  title: string;
}

export default function VideoCard({ videoId }: VideoCardProps): React.ReactElement {
  const [title, setTitle] = useState<string | null>(null);

  useEffect(() => {
    fetch(`https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`)
      .then((r) => r.json() as Promise<OEmbedResponse>)
      .then((d) => setTitle(d.title))
      .catch(() => setTitle("Watch on YouTube"));
  }, [videoId]);

  return (
    <a
      className={clsx(styles.card, "video-card")}
      href={`https://www.youtube.com/watch?v=${videoId}`}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={title ?? "YouTube video"}
    >
      <div className={styles.thumbnailWrapper}>
        <img src={`https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`} alt={title ?? ""} className={styles.thumbnail} loading="lazy" />
        <div className={styles.playOverlay} aria-hidden="true">
          <svg viewBox="0 0 68 48" className={styles.playIcon}>
            <path
              d="M66.5 7.7A8.5 8.5 0 0 0 60.7 2C55.4.5 34 .5 34 .5S12.6.5 7.3 2A8.5 8.5 0 0 0 1.5 7.7C0 13 0 24 0 24s0 11 1.5 16.3A8.5 8.5 0 0 0 7.3 46c5.3 1.5 26.7 1.5 26.7 1.5s21.4 0 26.7-1.5a8.5 8.5 0 0 0 5.8-5.7C68 35 68 24 68 24s0-11-1.5-16.3z"
              fill="currentColor"
            />
            <path d="M45 24 27 14v20z" fill="white" />
          </svg>
        </div>
      </div>
      <p className={styles.cardTitle}>{title ?? "\u00A0"}</p>
    </a>
  );
}
