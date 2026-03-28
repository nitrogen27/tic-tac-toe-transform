/**
 * Top-level page layout: header, board + sidebar, move history.
 */

import { Header } from "./Header";
import { Board } from "../Board/Board";
import { AnalysisSidebar } from "../Analysis/AnalysisSidebar";
import { GameControls } from "../Game/GameControls";
import { MoveHistory } from "../Game/MoveHistory";

export function MainLayout() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-4 p-4 lg:flex-row">
        {/* Left column: board + controls */}
        <div className="flex flex-1 flex-col items-center gap-4">
          <GameControls />
          <Board />
          <MoveHistory />
        </div>

        {/* Right column: analysis */}
        <AnalysisSidebar />
      </main>
    </div>
  );
}
