# Template: New React Page / Feature

## Шаги

1. Добавить типы в `api/types.ts` если нужны новые API-типы
2. Добавить API функцию в `api/client.ts`
3. Создать хук в `hooks/`
4. Создать компонент(ы) в `components/`
5. Добавить в layout (`components/Layout/MainLayout.tsx`)
6. Добавить тест в `tests/`

---

## API тип (api/types.ts)

```ts
export interface MyFeatureData {
  id: string;
  value: number;
  label: string;
}
```

## API функция (api/client.ts)

```ts
const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function fetchMyFeature(params: MyFeatureRequest): Promise<MyFeatureData> {
  const controller = new AbortController();
  const res = await fetch(`${BASE}/my-feature`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
    signal: controller.signal,
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json() as Promise<MyFeatureData>;
}
```

## Хук (hooks/useMyFeature.ts)

```ts
import { useState, useCallback } from "react";
import { fetchMyFeature } from "../api/client";
import type { MyFeatureData } from "../api/types";

export function useMyFeature() {
  const [data, setData] = useState<MyFeatureData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (params: MyFeatureRequest) => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchMyFeature(params);
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  return { data, loading, error, load };
}
```

## Компонент (components/MyFeature/MyFeaturePanel.tsx)

```tsx
import { useMyFeature } from "../../hooks/useMyFeature";
import { useGameState } from "../../store/gameStore";

export function MyFeaturePanel() {
  const { boardSize } = useGameState();
  const { data, loading, error, load } = useMyFeature();

  return (
    <div className="flex flex-col gap-3 p-4">
      {loading && <div className="text-gray-500 text-sm">Loading...</div>}
      {error && <div className="text-red-500 text-sm">{error}</div>}
      {data && (
        <div className="font-mono text-sm">
          {data.label}: {data.value.toFixed(3)}
        </div>
      )}
      <button
        className="px-4 py-2 rounded-md bg-board-line text-white hover:opacity-80"
        onClick={() => load({ boardSize })}
        disabled={loading}
      >
        Load
      </button>
    </div>
  );
}
```

## GameStore Action (если нужно сохранять в global state)

```ts
// В gameStore.tsx — добавить в GameAction:
| { type: "SET_MY_FEATURE"; data: MyFeatureData | null }

// В reducer:
case "SET_MY_FEATURE":
  return { ...state, myFeatureData: action.data };

// В GameState:
myFeatureData: MyFeatureData | null;
```

## Тест (tests/MyFeaturePanel.test.tsx)

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { MyFeaturePanel } from "../components/MyFeature/MyFeaturePanel";
import * as client from "../api/client";

vi.mock("../api/client");

test("shows data after load", async () => {
  vi.mocked(client.fetchMyFeature).mockResolvedValue({ id: "1", value: 0.5, label: "Test" });
  render(<GameProvider><MyFeaturePanel /></GameProvider>);
  await userEvent.click(screen.getByRole("button", { name: /load/i }));
  await waitFor(() => expect(screen.getByText("Test: 0.500")).toBeInTheDocument());
});
```

## Checklist

- [ ] Типы в `api/types.ts`
- [ ] Функция в `api/client.ts`
- [ ] Хук в `hooks/`
- [ ] Компонент в `components/`
- [ ] Добавлен в layout
- [ ] Тест написан
- [ ] Tailwind классы (не inline styles)
