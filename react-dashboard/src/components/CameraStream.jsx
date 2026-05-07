import { getVideoFeedUrl } from "../api";

function CameraStream() {
  return (
    <div className="overflow-hidden rounded-2xl border border-slate-700 bg-slate-900 shadow-xl">
      <div className="border-b border-slate-700 px-4 py-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Live Camera</h2>
      </div>
      <div className="aspect-video w-full bg-slate-800">
        <img src={getVideoFeedUrl()} alt="Raspberry Pi stream" className="h-full w-full object-cover" />
      </div>
    </div>
  );
}

export default CameraStream;
