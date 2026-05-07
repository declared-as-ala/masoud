function ConnectionStatus({ connected }) {
  return (
    <div
      className={`rounded-full border px-3 py-1 text-xs font-semibold ${
        connected
          ? "border-emerald-700 bg-emerald-500/20 text-emerald-300"
          : "border-rose-700 bg-rose-500/20 text-rose-300"
      }`}
    >
      {connected ? "Camera online" : "Camera offline"}
    </div>
  );
}

export default ConnectionStatus;
