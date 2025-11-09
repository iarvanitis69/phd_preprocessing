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
        print(f"total_duration: {total_duration:.2f}s")

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
    ordered = {}
    if "COUNT_OF_STATIONS" in data:
        ordered["COUNT_OF_STATIONS"] = data["COUNT_OF_STATIONS"]
    if "Events" in data:
        ordered["Events"] = data["Events"]
    for k, v in data.items():
        if k not in ordered:
            ordered[k] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)

def insert_record(db: dict, year: str, event_name: str, station: str, channel: str,
                  edge_max: float, middle_max: float, snr: float,
                  first_max: float, last_max: float, total_duration: float):
    if "Events" not in db:
        raise ValueError("Missing 'Events' in database.")
    if "COUNT_OF_STATIONS" not in db:
        db["COUNT_OF_STATIONS"] = 0

    events = db["Events"]
    if year not in events:
        events[year] = {}
    if event_name not in events[year]:
        events[year][event_name] = {}
    if station not in events[year][event_name]:
        events[year][event_name][station] = {}
        db["COUNT_OF_STATIONS"] += 1

    events[year][event_name][station][channel] = {
        "edge_max": edge_max,
        "middle_max": middle_max,
        "first_max": first_max,
        "last_max": last_max,
        "snr": snr,
        "total_duration": total_duration
    }

def is_excluded(excluded: dict, event: str, station: str) -> bool:
    return event in excluded and station in excluded[event]

def iter_year_dirs(base_dir):
    for year_dir in sorted(os.listdir(base_dir)):
        year_path = os.path.join(base_dir, year_dir)
        if os.path.isdir(year_path):
            yield year_dir, year_path

def iter_event_dirs(year_path):
    for event_dir in sorted(os.listdir(year_path)):
        event_path = os.path.join(year_path, event_dir)
        if os.path.isdir(event_path):
            yield event_dir, event_path

def iter_station_dirs(event_path):
    for station_dir in sorted(os.listdir(event_path)):
        station_path = os.path.join(event_path, station_dir)
        if os.path.isdir(station_path):
            if any(f.endswith("_demeanDetrend_IC.mseed") for f in os.listdir(station_path)):
                yield station_dir, station_path

def process_station(station_path, year, event_name, station_name, snr_db, excluded):
    if is_excluded(excluded, event_name, station_name):
        print(f"⛔ Παράλειψη excluded σταθμού: {event_name}/{station_name}")
        return

    channel_results = {}
    for filename in os.listdir(station_path):
        if not filename.endswith("_demeanDetrend_IC.mseed"):
            continue

        path = os.path.join(station_path, filename)
        try:
            st = read(path)
        except Exception as e:
            print(f"⚠️ Αποτυχία ανάγνωσης {path}: {e}")
            continue

        for tr in st:
            channel = getattr(tr.stats, "channel", "UNK") or "UNK"
            if channel in snr_db["Events"].get(year, {}).get(event_name, {}).get(station_name, {}):
                print(f"⏩ Ήδη υπάρχει: {event_name}/{station_name}/{channel} – Παράλειψη")
                continue

            try:
                edge_max, middle_max, first_max, last_max, total_duration = compute_edges_middle_max(tr)
                snr = compute_snr(edge_max, middle_max)
            except Exception as e:
                print(f"⚠️ Παράλειψη {path} {station_name}.{channel}: {e}")
                continue

            print(f"📊 {event_name}/{station_name}/{channel}: SNR={snr:.3f}")
            insert_record(snr_db, year, event_name, station_name, channel,
                          edge_max, middle_max, snr, first_max, last_max, total_duration)
            channel_results[channel] = snr

    required_channels = {"HHE", "HHN", "HHZ"}
    if required_channels.issubset(channel_results):
        min_snr = float(min(channel_results[ch] for ch in required_channels))
        snr_db["Events"][year][event_name][station_name]["minimum_snr"] = min_snr
        print(f"✅ Υπολογίστηκε minimum_snr για {event_name}/{station_name}: {min_snr:.3f}")

def find_snr():
    from main import BASE_DIR, LOG_DIR
    excluded = load_excluded_stations(LOG_DIR)
    snr_file = os.path.join(LOG_DIR, "snr.json")
    snr_db = load_json(snr_file)

    if "Events" not in snr_db:
        snr_db["Events"] = {}
    if "COUNT_OF_STATIONS" not in snr_db:
        snr_db["COUNT_OF_STATIONS"] = 0

    for year, year_path in iter_year_dirs(BASE_DIR):
        for event_name, event_path in iter_event_dirs(year_path):
            for station_name, station_path in iter_station_dirs(event_path):
                process_station(station_path, year, event_name, station_name, snr_db, excluded)
                save_json(snr_file, snr_db)

if __name__ == "__main__":
    find_snr()
