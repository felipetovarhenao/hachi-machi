import React, { useEffect, useRef, useState } from "react";
import styles from "./styles.module.css";

/**
 * YoutubePlaylist
 *
 * Props:
 *   videoIds: string[]   — ordered list of YouTube video IDs
 *   title?: string       — optional section heading
 *
 * Example usage in your landing page (index.tsx):
 *
 *   import YoutubePlaylist from "@site/src/components/YoutubePlaylist";
 *
 *   <YoutubePlaylist
 *     title="Video Tutorials"
 *     videoIds={["dQw4w9WgXcQ", "9bZkp7q19f0", "..."]}
 *   />
 */

interface VideoCardProps {
  videoId: string;
}

interface YoutubePlaylistProps {
  videoIds: string[];
  title?: string;
}

interface OEmbedResponse {
  title: string;
}

function VideoCard({ videoId }: VideoCardProps): React.ReactElement {
  const [title, setTitle] = useState<string | null>(null);

  // oEmbed gives us the title without an API key
  useEffect(() => {
    fetch(`https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`)
      .then((r) => r.json() as Promise<OEmbedResponse>)
      .then((d) => setTitle(d.title))
      .catch(() => setTitle("Watch on YouTube"));
  }, [videoId]);

  return (
    <a
      className={styles.card}
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

export default function YoutubePlaylist({ videoIds = [], title }: YoutubePlaylistProps): React.ReactElement | null {
  const trackRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  const updateScrollState = (): void => {
    const el = trackRef.current;
    if (!el) return;
    setCanScrollLeft(el.scrollLeft > 4);
    setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 4);
  };

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    updateScrollState();
    el.addEventListener("scroll", updateScrollState, { passive: true });
    const ro = new ResizeObserver(updateScrollState);
    ro.observe(el);
    return () => {
      el.removeEventListener("scroll", updateScrollState);
      ro.disconnect();
    };
  }, [videoIds]);

  const scroll = (dir: number): void => {
    const el = trackRef.current;
    if (!el) return;
    el.scrollBy({ left: dir * 520, behavior: "smooth" });
  };

  if (!videoIds.length) return null;

  return (
    <section className={styles.section}>
      {title && <h1 className={styles.heading}>{title.toUpperCase()}</h1>}
      <div className={styles.carousel}>
        {canScrollLeft && (
          <button className={`${styles.arrow} ${styles.arrowLeft}`} onClick={() => scroll(-1)} aria-label="Scroll left">
            ‹
          </button>
        )}
        <div className={styles.track} ref={trackRef}>
          {videoIds.map((id) => (
            <VideoCard key={id} videoId={id} />
          ))}
        </div>
        {canScrollRight && (
          <button className={`${styles.arrow} ${styles.arrowRight}`} onClick={() => scroll(1)} aria-label="Scroll right">
            ›
          </button>
        )}
      </div>
    </section>
  );
}
