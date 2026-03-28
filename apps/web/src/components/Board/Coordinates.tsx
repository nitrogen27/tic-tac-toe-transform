import { colLabel, rowLabel } from "../../utils/boardUtils";

interface CoordinatesProps {
  boardSize: number;
}

export function Coordinates({ boardSize }: CoordinatesProps) {
  const labels: JSX.Element[] = [];
  const offset = 0.65;

  for (let c = 0; c < boardSize; c++) {
    labels.push(
      <text
        key={`top-${c}`}
        x={c}
        y={-offset}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={0.35}
        fill="#5a4510"
      >
        {colLabel(c)}
      </text>
    );
    labels.push(
      <text
        key={`bot-${c}`}
        x={c}
        y={boardSize - 1 + offset}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={0.35}
        fill="#5a4510"
      >
        {colLabel(c)}
      </text>
    );
  }

  for (let r = 0; r < boardSize; r++) {
    const label = rowLabel(r, boardSize);
    labels.push(
      <text
        key={`left-${r}`}
        x={-offset}
        y={r}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={0.35}
        fill="#5a4510"
      >
        {label}
      </text>
    );
    labels.push(
      <text
        key={`right-${r}`}
        x={boardSize - 1 + offset}
        y={r}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={0.35}
        fill="#5a4510"
      >
        {label}
      </text>
    );
  }

  return <>{labels}</>;
}
