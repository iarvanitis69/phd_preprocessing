import os
import json
from obspy import read, Trace

def find_files_for_overlaps():
    overlaps = []

    from main import BASE_DIR
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

                mseed_path = os.path.join(station_path, "mseed")

                # Εδώ θα προσθέσουμε τον έλεγχο για overlaps στα mseed αρχεία
                overlaps_found = check_overlaps_in_mseed(mseed_path)

                if overlaps_found:
                    overlaps.append({
                        "year": year,
                        "event": event,
                        "station": station,
                        "overlaps": overlaps_found
                    })

    # Αποθήκευση σε JSON log
    from main import OVERLAPS_LOG_FILE
    with open(OVERLAPS_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(overlaps, f, indent=4, ensure_ascii=False)

    print(f"✅ Overlaps καταγράφηκαν στο: {OVERLAPS_LOG_FILE}")

def check_overlaps_in_mseed(mseed_path):
    """
    Ελέγχει για overlaps σε ΚΑΘΕ αρχείο .mseed ξεχωριστά.
    Επιστρέφει λίστα με overlaps (με λεπτομέρειες ανά αρχείο).
    """
    if not os.path.isdir(mseed_path):
        return []

    overlaps = []

    for fname in sorted(os.listdir(mseed_path)):
        if not fname.endswith(".mseed"):
            continue

        full_path = os.path.join(mseed_path, fname)
        print("Find overlaps in file:" + full_path)
        try:
            stream = read(full_path)

            # Δεν κάνουμε merge – ελέγχουμε κάθε trace/segment όπως είναι
            gaps = stream.get_gaps()

            for g in gaps:
                if g[6] < 0:   # Overlap
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