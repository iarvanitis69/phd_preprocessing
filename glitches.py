import os
import json
import numpy as np
from obspy import read, Trace
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_event_info(event_folder_name: str):
    try:
        parts = event_folder_name.split("_")
        origin_time = parts[0]
        lat = parts[1]
        lon = parts[2]
        depth = parts[3].replace("km", "")
        mag = parts[4].replace("M", "")
        return {
            "event_folder": event_folder_name,
            "origin_time": origin_time,
            "latitude": float(lat),
            "longitude": float(lon),
            "depth_km": float(depth),
            "magnitude": float(mag)
        }
    except Exception:
        return {
            "event_folder": event_folder_name,
            "origin_time": None,
            "latitude": None,
            "longitude": None,
            "depth_km": None,
            "magnitude": None
        }

def find_glitches(trace: Trace, threshold, window_size: int = 2):
    data = trace.data.astype(float)
    max_val = np.max(np.abs(data))
    if max_val == 0:
        return []

    data /= max_val
    glitches = []

    for i in range(len(data) - 2 * window_size):
        win1 = data[i:i + window_size]
        win2 = data[i + window_size:i + 2 * window_size]
        d1 = np.diff(win1)
        d2 = np.diff(win2)

        peak_rise = np.max(d1)
        peak_fall = np.min(d2)

        if peak_rise > threshold and peak_fall < -threshold:
            glitches.append({
                "start_index": i,
                "end_index": i + 2 * window_size,
                "channel": trace.stats.channel,
                "station": trace.stats.station,
                "network": trace.stats.network,
                "start_time": str(trace.stats.starttime + i / trace.stats.sampling_rate),
                "end_time": str(trace.stats.starttime + (i + 2 * window_size) / trace.stats.sampling_rate),
                "peak_rise": round(peak_rise, 4),
                "peak_fall": round(peak_fall, 4)
            })

    return glitches

def find_files_for_glitches_parallel(threshold: float = 1.0, max_workers: int = 4):
    from main import BASE_DIR
    print(f"🚀 Έναρξη σάρωσης στο: {BASE_DIR}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for year in sorted(os.listdir(BASE_DIR)):
            year_path = os.path.join(BASE_DIR, year)
            if not os.path.isdir(year_path) or year == "Logs":
                continue

            for event in sorted(os.listdir(year_path)):
                event_path = os.path.join(year_path, event)
                if not os.path.isdir(event_path):
                    continue

                futures.append(executor.submit(process_single_event, event_path, threshold))

        for f in as_completed(futures):
            print(f.result())

    print("💾 Όλα τα glitches έχουν ήδη γραφτεί στο Logs/glitches.json.")

def process_single_event(event_path: str, threshold: float):
    from main import BASE_DIR
    logs_dir = os.path.join(BASE_DIR, "Logs")
    excluded = load_excluded_stations(logs_dir)

    event_info = extract_event_info(os.path.basename(event_path))
    event_name = event_info["event_folder"]

    for root, _, files in os.walk(event_path):
        for f in files:
            if not f.endswith(".mseed"):
                continue

            full_path = os.path.join(root, f)

            try:
                st = read(full_path)
            except Exception as e:
                print(f"❌ Αποτυχία ανάγνωσης {full_path}: {e}")
                continue

            total_glitches = 0
            for tr in st:
                network = tr.stats.network
                station = tr.stats.station
                net_station = f"{network}.{station}"

                if event_name in excluded and net_station in excluded[event_name]:
                    print(f"⏭️ Παράλειψη excluded station: {event_name}/{net_station}")
                    continue

                glitches = find_glitches(tr, threshold=threshold)
                if glitches:
                    total_glitches += len(glitches)
                    for g in glitches:
                        g["file"] = os.path.basename(full_path)
                    append_to_json_file(
                        event_name=event_name,
                        station=station,
                        channel=tr.stats.channel,
                        glitches=glitches
                    )
                    print(f"📌 {event_name} | {tr.id} | {len(glitches)} glitches")

            if total_glitches > 0:
                update_excluded_stations(event_name, net_station, "Glitches", logs_dir)

    return f"✅ Ολοκληρώθηκε: {event_name}"

def process_mseed_file(mseed_path: str, event_name: str, threshold: float):
    try:
        st = read(mseed_path)
    except Exception as e:
        print(f"❌ Αποτυχία ανάγνωσης {mseed_path}: {e}")
        return

    for tr in st:
        glitches = find_glitches(tr, threshold=threshold)
        if glitches:
            for g in glitches:
                g["file"] = os.path.basename(mseed_path)
            append_to_json_file(
                event_name=event_name,
                station=tr.stats.station,
                channel=tr.stats.channel,
                glitches=glitches
            )
            print(f"📌 {event_name} | {tr.id} | {len(glitches)} glitches")

def append_to_json_file(event_name, station, channel, glitches):
    from main import BASE_DIR
    logs_path = os.path.join(BASE_DIR, "Logs")
    os.makedirs(logs_path, exist_ok=True)
    output_file = os.path.join(logs_path, "glitches.json")

    import multiprocessing
    lock = multiprocessing.Manager().Lock()
    with lock:
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
               try:
                   data = json.load(f)
               except json.JSONDecodeError:
                   data = {}
        else:
            data = {}

        net = glitches[0].get("network", "XX")
        net_station_key = f"{net}.{station}"

        if event_name not in data:
            data[event_name] = {}
        if net_station_key not in data[event_name]:
            data[event_name][net_station_key] = {}
        if channel not in data[event_name][net_station_key]:
            data[event_name][net_station_key][channel] = {"count": 0, "glitches": []}

        data[event_name][net_station_key][channel]["glitches"].extend(glitches)
        data[event_name][net_station_key][channel]["count"] += len(glitches)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def load_excluded_stations(logs_dir: str):
    excluded_path = os.path.join(logs_dir, "excluded_stations.json")
    if os.path.exists(excluded_path):
        with open(excluded_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def update_excluded_stations(event_name: str, net_station: str, reason: str, logs_dir: str):
    excluded_path = os.path.join(logs_dir, "excluded_stations.json")
    excluded = load_excluded_stations(logs_dir)

    if event_name not in excluded:
        excluded[event_name] = {}

    if net_station not in excluded[event_name]:
        excluded[event_name][net_station] = {"reason": reason}
        excluded["COUNT"] += 1
    else:
        excluded[event_name][net_station]["reason"] = reason

    with open(excluded_path, "w", encoding="utf-8") as f:
        json.dump(excluded, f, indent=2, ensure_ascii=False)
