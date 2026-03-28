export function Header() {
  return (
    <header className="border-b border-neutral-200 bg-white px-6 py-3">
      <div className="mx-auto flex max-w-7xl items-center justify-between">
        <h1 className="text-lg font-bold tracking-tight text-neutral-800">
          Gomoku Platform
        </h1>
        <span className="text-xs text-neutral-400">v0.1.0</span>
      </div>
    </header>
  );
}
