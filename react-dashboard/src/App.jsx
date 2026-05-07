import { useEffect, useState } from "react";
import { api } from "./api";
import CameraStream from "./components/CameraStream";
import DetectionCards from "./components/DetectionCards";
import HumanStatus from "./components/HumanStatus";
import AlertsList from "./components/AlertsList";
import ConnectionStatus from "./components/ConnectionStatus";

function App() {
  const [detections, setDetections] = useState({
    timestamp: "",
    objects: [],
    human: { detected: false, authorized: null, name: null },
    last_alert: null,
  });
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const fetchDetections = async () => {
      try {
        const { data } = await api.get("/detections");
        setDetections(data);
        setConnected(true);
      } catch {
        setConnected(false);
      }
    };

    const fetchAlerts = async () => {
      try {
        const { data } = await api.get("/alerts");
        setAlerts(data?.alerts || []);
      } catch {
        // Keep old alerts on transient failure.
      }
    };

    fetchDetections();
    fetchAlerts();
    const detectionTimer = window.setInterval(fetchDetections, 1000);
    const alertsTimer = window.setInterval(fetchAlerts, 5000);

    return () => {
      window.clearInterval(detectionTimer);
      window.clearInterval(alertsTimer);
    };
  }, []);

  const tomatoDetected = detections.objects.some((x) => x.label === "tomato");
  const pepperDetected = detections.objects.some((x) => x.label === "pepper" || x.label === "felfel");
  const unauthorized = detections.human?.detected && detections.human?.authorized === false;

  return (
    <div className="min-h-screen bg-slate-950 p-6 text-slate-100">
      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-6 lg:grid-cols-12">
        <section className="lg:col-span-8">
          <div className="mb-4 flex items-center justify-between">
            <h1 className="text-2xl font-semibold">Raspberry Pi Vision Dashboard</h1>
            <ConnectionStatus connected={connected} />
          </div>
          <CameraStream />
          <div className="mt-4 rounded-xl border border-slate-700 bg-slate-900/60 p-4">
            <p className="text-xs uppercase tracking-wide text-slate-400">Last update</p>
            <p className="text-sm text-slate-200">{detections.timestamp || "Waiting for data..."}</p>
            <p className="mt-2 text-sm text-slate-300">
              Tomato: <b>{tomatoDetected ? "Yes" : "No"}</b> | Pepper: <b>{pepperDetected ? "Yes" : "No"}</b>
            </p>
          </div>
          <div className="mt-4">
            <DetectionCards detections={detections} />
          </div>
        </section>

        <aside className="space-y-4 lg:col-span-4">
          {unauthorized ? (
            <div className="rounded-xl border border-rose-700 bg-rose-500/15 p-4">
              <p className="text-sm font-semibold text-rose-300">Unauthorized human detected</p>
              <p className="mt-1 text-xs text-rose-200">Telegram alert is sent with cooldown protection.</p>
            </div>
          ) : null}
          <HumanStatus human={detections.human} />
          <AlertsList alerts={alerts} />
        </aside>
      </div>
    </div>
  );
}

export default App;
