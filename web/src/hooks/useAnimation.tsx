import { useEffect, useRef } from "react";

export default function useCSSVarAnimation(
  varName: string,
  getValue: (elapsed: number) => number,
  unit: string = "",
): React.RefObject<HTMLElement | null> {
  const elementRef = useRef<HTMLElement>(null);
  const animRef = useRef<number>(null);

  useEffect(() => {
    const startTime = performance.now();
    const tick = () => {
      const value = getValue(performance.now() - startTime);
      elementRef.current?.style.setProperty(varName, `${value}${unit}`);
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current!);
  }, [varName, getValue, unit]);

  return elementRef;
}
