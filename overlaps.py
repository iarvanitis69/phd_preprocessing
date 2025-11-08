import os
import json
from obspy import read, Trace

def load_excluded_stations(logs_dir):
    """Φορτώνει το excluded_stations.json ως λεξικό {event: {station: {reason}}}"""
    excluded_path = os.path.join(logs_dir, "excluded_stations.json")
    if os.path.exists(excluded_path):
        with open(excluded_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def update_excluded_stations(event, station, reason, logs_dir):
    """Ενημερώνει το excluded_stations.json με νέο event/station/reason αν δεν υπάρχει ήδη."""
    path = os.path.join(logs_dir, "excluded_stations.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    if event not in data:
        data[event] = {}

    if station not in data[event]:
        data[event][station] = {"reason": reason}
        data["COUNT"] += 1
    else:
        # Αν υπάρχει ήδη, μην τον ξαναγράψεις με διαφορετικό reason
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"📌 Προστέθηκε στους excluded: {event}/{station} → {reason}")

def find_files_for_overlaps():
    overlaps = []

    from main import BASE_DIR, OVERLAPS_LOG_FILE
    logs_dir = os.path.join(BASE_DIR, "Logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 🔹 Φόρτωσε τους excluded σταθμούς
    excluded = load_excluded_stations(logs_dir)

    for year in sorted(os.listdir(BASE_DIR)):
        year_path = os.path.join(BASE_DIR, year)
        if not os.path.isdir(year_path) or year == "Logs":
            continue

        for event in sorted(os.listdir(year_path)):
            event_path = os.path.join(year_path, event)
            if not os.path.isdir(event_path):
                continue

            for station in sorted(os.listdir(event_path)):
                station_path = os.path.join(event_path, station)
                if not os.path.isdir(station_path) or station.lower() == "info.txt":
                    continue

                # ✅ Αν ήδη excluded → skip
                if event in excluded and station in excluded[event]:
                    print(f"⏭️ Παράλειψη excluded station: {event}/{station}")
                    continue

                mseed_path = os.path.join(station_path, "mseed")
                overlaps_found = check_overlaps_in_mseed(mseed_path)

                if overlaps_found:
                    overlaps.append({
                        "year": year,
                        "event": event,
                        "station": station,
                        "overlaps": overlaps_found
                    })

                    # ✅ Πρόσθεσέ τον στο excluded_stations.json με reason "Overlaps"
                    update_excluded_stations(event, station, "Overlaps", logs_dir)

    with open(OVERLAPS_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(overlaps, f, indent=4, ensure_ascii=False)

    print(f"✅ Overlaps καταγράφηκαν στο: {OVERLAPS_LOG_FILE}")

def check_overlaps_in_mseed(mseed_path):
    if not os.path.isdir(mseed_path):
        return []

    overlaps = []

    for fname in sorted(os.listdir(mseed_path)):
        if not fname.endswith(".mseed"):
            continue

        full_path = os.path.join(mseed_path, fname)
        print("🔍 Find overlaps in file: " + full_path)

        try:
            stream = read(full_path)
            gaps = stream.get_gaps()

            for g in gaps:
                if g[6] < 0:  # ✅ Overlap
                    overlaps.append({
                        "file": fname,
                        "network": g[0],
                        "station": g[1],
                        "location": g[2],
                        "channel": g[3],
                        "start": str(g[4]),
                        "end": str(g[5]),
                        "duration_sec": abs(g[6])
                    })

        except Exception as e:
            overlaps.append({
                "file": fname,
                "error": str(e)
            })

    return overlaps
