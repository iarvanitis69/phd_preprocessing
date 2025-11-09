import os
import json
import numpy as np
from obspy import read, Trace

PRESET_SEC = 30
END_SEC = 30

def load_excluded_stations(logs_dir):
    path = os.path.join(logs_dir, "excluded_stations.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def normalize_array(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    maxabs = np.max(np.abs(arr)) if arr.size else 0.0
    if maxabs == 0.0:
        return arr
    return arr / maxabs

def compute_edges_middle_max(trace: Trace):
    fs = float(trace.stats.sampling_rate)
    data = trace.data.astype(float)

    total_duration = len(data) / fs if fs > 0 else 0.0
    if total_duration < 190:
        print(f"total_duration:{total_duration:.2f}s")

    w1 = int(PRESET_SEC * fs)
    w2 = int((total_duration - PRESET_SEC - END_SEC) * fs)
    w3 = int(END_SEC * fs)

    win1 = data[:w1]
    win2 = data[w1:w1 + w2]
    win3 = data[w1 + w2:w1 + w2 + w3]

    edge_combined = np.concatenate([win1, win3]) if (win1.size + win3.size) else np.array([])

    edge_max = float(np.max(np.abs(edge_combined))) if edge_combined.size else 0.0
    first_max = float(np.max(np.abs(win1))) if win1.size else 0.0
    last_max = float(np.max(np.abs(win3))) if win3.size else 0.0
    middle_max = float(np.max(np.abs(win2))) if win2.size else 0.0

    return edge_max, middle_max, first_max, last_max, total_duration

def compute_snr(edge_max: float, middle_max: float) -> float:
    eps = 1e-12
    return float(middle_max / (edge_max + eps))

def load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def insert_record(db: dict, event_name: str, station: str, channel: str,
                  edge_max: float, middle_max: float, snr: float,
                  first_max: float, last_max: float, total_duration: float):
    ev = db.setdefault(event_name, {})
    st = ev.setdefault(station, {})
    ch = st.setdefault(channel, {})
    ch.update({
        "edge_max": edge_max,
        "middle_max": middle_max,
        "first_max": first_max,
        "last_max": last_max,
        "snr": snr,
        "total_duration": total_duration
    })

def guess_event_name(file_path: str) -> str:
    parts = os.path.normpath(file_path).split(os.sep)
    if "Events" in parts:
        idx = parts.index("Events")
        if idx + 2 < len(parts):
            return parts[idx + 2]
    return os.path.splitext(os.path.basename(file_path))[0]

def is_excluded(excluded: dict, event: str, station: str) -> bool:
    return event in excluded and station in excluded[event]


def process_file(path: str, snr_file_path: str, excluded: dict):
    db_snr = load_json(snr_file_path)
    if "count" not in db_snr:
        db_snr["count"] = 0

    event_name = guess_event_name(path)

    try:
        st = read(path)
    except Exception as e:
        print(f"⚠️ Αποτυχία ανάγνωσης {path}: {e}")
        return

    for tr in st:
        network = getattr(tr.stats, "network", "XX") or "XX"
        station_code = getattr(tr.stats, "station", "UNK") or "UNK"
        station = f"{network}.{station_code}"
        channel = getattr(tr.stats, "channel", "UNK") or "UNK"

        if is_excluded(excluded, event_name, station):
            print(f"⛔ Παράλειψη excluded σταθμού: {event_name}/{station}/{channel}")
            continue

        if (event_name in db_snr and
                station in db_snr[event_name] and
                channel in db_snr[event_name][station]):
            print(f"⏩ Ήδη υπάρχει: {event_name}/{station}/{channel} – Παράλειψη")
            continue

        try:
            edge_max, middle_max, first_max, last_max, total_duration = compute_edges_middle_max(tr)
            snr = compute_snr(edge_max, middle_max)
        except Exception as e:
            print(f"⚠️ Παράλειψη {path} {station}.{channel}: {e}")
            continue

        print(f"📊 {event_name}/{station}/{channel}: SNR={snr:.3f}")

        insert_record(db_snr, event_name, station, channel,
                      edge_max, middle_max, snr, first_max, last_max, total_duration)

        # ✅ Υπολογισμός minimum_snr μόνο όταν υπάρχουν και τα 3 απαραίτητα κανάλια
        required_channels = {"HHE", "HHN", "HHZ"}
        station_channels = db_snr[event_name][station]
        available_channels = {ch for ch in station_channels if
                              isinstance(station_channels[ch], dict) and "snr" in station_channels[ch]}

        if required_channels.issubset(available_channels):
            snrs = [station_channels[ch]["snr"] for ch in required_channels]
            min_snr = float(min(snrs))
            db_snr[event_name][station]["minimum_snr"] = min_snr
            print(f"✅ Υπολογίστηκε minimum_snr για {event_name}/{station}: {min_snr:.3f}")

        db_snr["count"] += 1
        save_json(snr_file_path, db_snr)


def iter_mseed_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        if "Logs" in dirpath:
            continue
        for fn in filenames:
            if fn.endswith("_demeanDetrend_IC.mseed"):
                yield os.path.join(dirpath, fn)

def find_snr():
    from main import BASE_DIR, LOG_DIR
    excluded = load_excluded_stations(LOG_DIR)
    snr_file = os.path.join(LOG_DIR, "snr.json")
    for f in iter_mseed_files(BASE_DIR):
        process_file(f, snr_file, excluded)

if __name__ == "__main__":
    find_snr()
