#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Βήμα 1 – Demean + Detrend → *_demeanDetrend.mseed (INT32 / Steim2)

Διαβάζει τα raw MiniSEED αρχεία (Steim2), εφαρμόζει:
  1️⃣ demean
  2️⃣ detrend
και αποθηκεύει τα αποτελέσματα πάλι σε Steim2 (encoding=11, int32).

✔ Απόλυτη συμβατότητα με Obspy / evalresp
✔ Καμία απώλεια δεδομένων
✔ Καθαρή καταγραφή σφαλμάτων ανά event/station/channel
✔ Παράκαμψη excluded σταθμών από το Logs/excluded_stations.json
"""

import os, json, numpy as np
from obspy import read, Stream

#BASE_DIR = "/media/iarv/Samsung"
#EVENTS_DIR = os.path.join(BASE_DIR, "Events")
#LOGS_DIR = os.path.join(BASE_DIR, "Logs")
#os.makedirs(LOGS_DIR, exist_ok=True)
# ERROR_PATH = os.path.join(LOGS_DIR, "demeanDetrend_errors.json")
# EXCLUDED_PATH = os.path.join(LOGS_DIR, "excluded_stations.json")


def load_excluded_stations():
    from main import LOG_DIR
    EXCLUDED_PATH = os.path.join(LOG_DIR, "excluded_stations.json")
    if os.path.exists(EXCLUDED_PATH):
        try:
            with open(EXCLUDED_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except json.JSONDecodeError:
            pass
    return {}

def get_excluded():
    """Φορτώνει πάντα την τρέχουσα έκδοση του excluded_stations.json."""
    return load_excluded_stations()


def log_error(year, event, station, filename, msg):
    from instrument_correction import LOGS_DIR
    ERROR_PATH = os.path.join(LOGS_DIR, "demeanDetrend_errors.json")
    data = {}
    if os.path.exists(ERROR_PATH):
        try:
            with open(ERROR_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass
    data.setdefault(year, {}).setdefault(event, {}).setdefault(station, []).append(f"{filename}: {msg}")
    with open(ERROR_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"🛑 {year}/{event}/{station}/{filename} → {msg}")

def is_station_excluded(event_name, station):
    excluded = get_excluded()
    if event_name in excluded:
        for net_station in excluded[event_name]:
            if "." in net_station:
                _, excluded_station = net_station.split(".")
                if excluded_station == station:
                    return True
    return False

def process_station_dir(station_dir, year, event):
    station = os.path.basename(station_dir)
    if is_station_excluded(event, station):
        msg = "Παράλειψη: Σταθμός έχει σημειωθεί ως excluded (π.χ. λιγότερα από 3 κανάλια)"
        print(f"🚫 Παράκαμψη excluded σταθμού: {event}/{station}")
        log_error(year, event, station, "-", msg)
        return

    files = [f for f in os.listdir(station_dir)
             if f.endswith(".mseed") and "_demeanDetrend" not in f]

    for fname in sorted(files):
        in_path = os.path.join(station_dir, fname)
        out_path = in_path.replace(".mseed", "_demeanDetrend.mseed")

        if os.path.exists(out_path):
            print(f"⏩ Παράκαμψη: {out_path}")
            continue

        try:
            st = read(in_path)
        except Exception as e:
            log_error(year, event, station, fname, f"Ανάγνωση: {e}")
            continue

        traces = []
        for tr in st:
            try:
                tr.detrend("demean")
                tr.detrend("linear")
                tr.data = np.nan_to_num(tr.data, nan=0, posinf=0, neginf=0)

                max_val = np.max(np.abs(tr.data))
                scale = 1e6 / max_val if max_val != 0 else 1.0
                tr.data = np.ascontiguousarray((tr.data * scale).astype(np.int32))

                if hasattr(tr.stats, "mseed") and "encoding" in tr.stats.mseed:
                    tr.stats.mseed.encoding = None

                traces.append(tr)

            except Exception as e:
                log_error(year, event, station, fname, f"Επεξεργασία {tr.id}: {e}")

        if not traces:
            log_error(year, event, station, fname, "Κενό μετά το demean/detrend")
            continue

        try:
            st_out = Stream(traces)
            for tr in st_out:
                if hasattr(tr.stats, "mseed") and "encoding" in tr.stats.mseed:
                    tr.stats.mseed.encoding = None
            st_out.write(out_path, format="MSEED", encoding=11, reclen=4096)
            print(f"✅ Αποθηκεύτηκε (INT32 / Steim2): {out_path}")
        except Exception as e:
            log_error(year, event, station, fname, f"Αποθήκευση: {e}")


def demean_detrend():
    from main import BASE_DIR
    for year in sorted(os.listdir(BASE_DIR)):
        ydir = os.path.join(BASE_DIR, year)
        if not os.path.isdir(ydir):
            continue
        for event in sorted(os.listdir(ydir)):
            edir = os.path.join(ydir, event)
            if not os.path.isdir(edir):
                continue
            for station in sorted(os.listdir(edir)):
                sdir = os.path.join(edir, station)
                if os.path.isdir(sdir):
                    process_station_dir(sdir, year, event)


if __name__ == "__main__":
    demean_detrend()
