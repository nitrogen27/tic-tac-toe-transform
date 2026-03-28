/**
 * Visual bar showing position evaluation from -1 (white) to +1 (black).
 */

interface ValueBarProps {
  value: number; // -1..1
  confidence: number; // 0..1
}

export function ValueBar({ value, confidence }: ValueBarProps) {
  // Map [-1, 1] to [0%, 100%] where 50% is even.
  const pct = ((value + 1) / 2) * 100;

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-neutral-500">
        <span>White</span>
        <span>
          {value >= 0 ? "+" : ""}
          {value.toFixed(3)} ({(confidence * 100).toFixed(1)}%)
        </span>
        <span>Black</span>
      </div>
      <div className="h-3 w-full overflow-hidden rounded bg-neutral-200">
        <div
          className="h-full rounded bg-neutral-800 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
