import { GameProvider } from "./store/gameStore";
import { MainLayout } from "./components/Layout/MainLayout";

export default function App() {
  return (
    <GameProvider>
      <MainLayout />
    </GameProvider>
  );
}
