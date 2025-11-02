#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Διαβάζει όλα τα MiniSEED αρχεία από υποφακέλους mseed/ κάτω από το BASE_DIR
και γράφει συγκεντρωτικά αποτελέσματα σε BASE_DIR/Logs/snrlt5.json
ΜΟΝΟ όταν SNR < 5.
Δεν τροποποιεί τα πρωτότυπα αρχεία.
"""

import os
import json
import shutil

import numpy as np
from obspy import read, Trace

# === ΡΥΘΜΙΣΕΙΣ ===
BASE_DIR = "/media/iarv/Samsung/Events"
LOG_DIR = os.path.join(BASE_DIR, "Logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "snrlt5.json")

PRESET_SEC = 30
MIDDLE_SEC = 150
END_SEC = 30
TOTAL_SEC = PRESET_SEC + MIDDLE_SEC + END_SEC

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
    if total_duration < TOTAL_SEC:
        raise ValueError(f"Σήμα μικρότερο από {TOTAL_SEC}s: {total_duration:.2f}s")

    data = normalize_array(data)

    w1 = int(PRESET_SEC * fs)
    w2 = int(MIDDLE_SEC * fs)
    w3 = int(END_SEC * fs)

    win1 = data[:w1]
    win2 = data[w1:w1 + w2]
    win3 = data[w1 + w2:w1 + w2 + w3]

    edge_combined = np.concatenate([win1, win3]) if (win1.size + win3.size) else np.array([])

    edge_max = float(np.max(np.abs(edge_combined))) if edge_combined.size else 0.0
    first_max = float(np.max(np.abs(win1))) if win1.size else 0.0
    last_max = float(np.max(np.abs(win3))) if win3.size else 0.0
    middle_max = float(np.max(np.abs(win2))) if win2.size else 0.0

    return edge_max, middle_max, first_max, last_max

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
                  edge_max: float, middle_max: float, snr: float, first_max: float, last_max: float):
    ev = db.setdefault(event_name, {})
    st = ev.setdefault(station, {})
    ch = st.setdefault(channel, {})
    ch.update({
        "edge_max": edge_max,
        "middle_max": middle_max,
        "first_max": first_max,
        "last_max": last_max,
        "snr": snr
    })

def guess_event_name(file_path: str) -> str:
    parts = os.path.normpath(file_path).split(os.sep)
    if "Events" in parts:
        idx = parts.index("Events")
        if idx + 2 < len(parts):
            return parts[idx + 2]  # π.χ. Events/2010/<EVENT>
    return os.path.splitext(os.path.basename(file_path))[0]

def process_file(path: str):
    try:
        st = read(path)
    except Exception as e:
        print(f"⚠️ Αποτυχία ανάγνωσης {path}: {e}")
        return

    # φόρτωση δύο logs
    db_bad = load_json(LOG_FILE)  # snrlt5.json
    db_good = load_json(os.path.join(LOG_DIR, "snr.json"))

    if "count" not in db_bad:
        db_bad["count"] = 0
    if "count" not in db_good:
        db_good["count"] = 0

    event_name = guess_event_name(path)

    event_has_low_snr = False
    event_has_all_good = True
    station_snr_map = {}

    for tr in st:
        station = getattr(tr.stats, "station", "UNK") or "UNK"
        channel = getattr(tr.stats, "channel", "UNK") or "UNK"

        try:
            edge_max, middle_max, first_max, last_max = compute_edges_middle_max(tr)
            snr = compute_snr(edge_max, middle_max)
        except Exception as e:
            print(f"⚠️ Παράλειψη {path} {station}.{channel}: {e}")
            continue

        station_entry = station_snr_map.setdefault(station, {})
        station_entry[channel] = {
            "edge_max": edge_max,
            "middle_max": middle_max,
            "first_max": first_max,
            "last_max": last_max,
            "snr": snr
        }

        if snr < 5.0:
            event_has_low_snr = True
            event_has_all_good = False
            print(f"❌ {event_name}/{station}/{channel}: SNR={snr:.3f} < 5")
        else:
            print(f"✅ {event_name}/{station}/{channel}: SNR={snr:.3f} ≥ 5")

    # --- Κατηγοριοποίηση ---
    for station, channels in station_snr_map.items():
        snr_values = [c["snr"] for c in channels.values()]
        if all(v >= 5 for v in snr_values):
            # όλοι οι δίαυλοι έχουν SNR >= 5 → snr.json
            for ch, vals in channels.items():
                insert_record(db_good, event_name, station, ch,
                              vals["edge_max"], vals["middle_max"],
                              vals["snr"], vals["first_max"], vals["last_max"])
            db_good["count"] += 1
        else:
            # τουλάχιστον ένας δίαυλος SNR < 5 → snrlt5.json
            for ch, vals in channels.items():
                if vals["snr"] < 5:
                    insert_record(db_bad, event_name, station, ch,
                                  vals["edge_max"], vals["middle_max"],
                                  vals["snr"], vals["first_max"], vals["last_max"])
            db_bad["count"] += 1

    save_json(LOG_FILE, db_bad)
    save_json(os.path.join(LOG_DIR, "snr.json"), db_good)


def iter_mseed_files(root: str):
    """
    Επιστρέφει όλα τα αρχεία *_demean_detrend_IC.mseed
    μέσα στη δομή Events/<Year>/<Event>/<Station>/
    """
    for dirpath, _, filenames in os.walk(root):
        # Παράλειψε το Logs/
        if "Logs" in dirpath:
            continue

        for fn in filenames:
            if fn.endswith("_demean_detrend_IC.mseed"):
                yield os.path.join(dirpath, fn)

def find_snr():
    for f in iter_mseed_files(BASE_DIR):
        process_file(f)

def find_event_path(event_name):
    """
    Επιστρέφει το πλήρες path του event ψάχνοντας μέσα στα έτη (π.χ. /Events/2010/EVENT_NAME)
    """
    for year in os.listdir(BASE_DIR):
        year_path = os.path.join(BASE_DIR, year)
        if not os.path.isdir(year_path):
            continue
        candidate = os.path.join(year_path, event_name)
        if os.path.isdir(candidate):
            return candidate
    return None


def delete_stations_with_snr_lt5():
    """
    Διαγράφει κάθε φάκελο σταθμού (station) που έχει έστω ένα κανάλι με SNR < 5
    σύμφωνα με το αρχείο Logs/snrlt5.json.
    Αν μετά δεν μείνει κανένας σταθμός, διαγράφεται και ο φάκελος του event.
    """
    if not os.path.exists(LOG_FILE):
        print(f"❌ Δεν βρέθηκε το αρχείο {LOG_FILE}")
        return

    # Φόρτωση JSON
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ Σφάλμα ανάγνωσης JSON: {e}")
        return

    if not isinstance(data, dict) or len(data) <= 1:
        print("⚠️ Δεν υπάρχουν events ή σταθμοί με SNR < 5.")
        return

    deleted_stations = []
    deleted_events = []
    skipped_stations = []

    for event_name, event_data in data.items():
        if event_name == "count":
            continue

        # Βρες σωστά τον φάκελο του event
        event_path = find_event_path(event_name)
        if not event_path:
            print(f"⚠️ Δεν βρέθηκε φάκελος για το event {event_name}")
            continue

        stations_to_delete = []
        for station_name, station_channels in event_data.items():
            # Ελέγχουμε αν υπάρχει κανάλι με SNR < 5
            low_snr_found = any(
                (isinstance(ch_data, dict) and ch_data.get("snr", 9999) < 5)
                for ch_data in station_channels.values()
            )
            if low_snr_found:
                stations_to_delete.append(station_name)
            else:
                skipped_stations.append(f"{event_name}/{station_name}")

        # Διαγραφή σταθμών
        for station_name in stations_to_delete:
            station_path = os.path.join(event_path, station_name)
            if os.path.exists(station_path):
                try:
                    shutil.rmtree(station_path)
                    deleted_stations.append(station_path)
                    print(f"🗑️ Διαγράφηκε σταθμός: {station_path}")
                except Exception as e:
                    print(f"⚠️ Σφάλμα διαγραφής {station_path}: {e}")
            else:
                print(f"⚠️ Δεν βρέθηκε ο φάκελος σταθμού: {station_path}")

        # Αν δεν έχει μείνει κανένας σταθμός → διαγραφή event
        remaining = [
            d for d in os.listdir(event_path)
            if os.path.isdir(os.path.join(event_path, d))
        ]
        if len(remaining) == 0:
            try:
                shutil.rmtree(event_path)
                deleted_events.append(event_path)
                print(f"📁 Διαγράφηκε και το κενό event: {event_path}")
            except Exception as e:
                print(f"⚠️ Σφάλμα διαγραφής event {event_path}: {e}")

    print("\n=== ΑΠΟΤΕΛΕΣΜΑΤΑ ===")
    print(f"✅ Διαγράφηκαν {len(deleted_stations)} σταθμοί με SNR < 5.")
    print(f"📂 Διαγράφηκαν {len(deleted_events)} events χωρίς σταθμούς.")
    print(f"⏭️ Παραλείφθηκαν {len(skipped_stations)} σταθμοί χωρίς χαμηλό SNR.")
    print("=====================\n")


# --- Εκτέλεση συνάρτησης ---
if __name__ == "__main__":
    find_snr()
    #delete_stations_with_snr_lt5()