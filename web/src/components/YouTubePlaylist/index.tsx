import React, { useEffect, useRef, useState } from "react";
import styles from "./styles.module.css";
import VideoCard from "@site/src/components/VideoCard";

interface YoutubePlaylistProps {
  videoIds: string[];
  title?: string;
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
