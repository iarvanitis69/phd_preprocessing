#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FindEventStart
--------------
Για κάθε αρχείο *_demean_detrend_IC.mseed εντοπίζει τη χρονική στιγμή έναρξης
του σεισμικού κύματος χρησιμοποιώντας τον αλγόριθμο AIC.

Αποτελέσματα:
  /media/iarv/Samsung/Logs/event_startpoints.json
με δομή:
{
  "<EventName>": {
    "<StationName>": {
      "<ChannelName>": {
        "start_sample": <int>,
        "start_time": "<UTCDateTime>",
        "aic_min_value": <float>
      }
    }
  }
}
"""

import os
import json
import numpy as np
from obspy import read, UTCDateTime

# ---------------------------------------------------------
# Διαχείριση JSON
# ---------------------------------------------------------
def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def insert_result(db, event, station, channel, start_sample, start_time, aic_min):
    ev = db.setdefault(event, {})
    st = ev.setdefault(station, {})
    ch = st.setdefault(channel, {})
    ch.update({
        "start_sample": start_sample,
        "start_time": str(start_time),
        "aic_min_value": aic_min
    })

# ---------------------------------------------------------
# Κύρια συνάρτηση εντοπισμού έναρξης
# ---------------------------------------------------------
def find_event_start():
    from main import LOG_DIR, BASE_DIR
    OUTPUT_JSON = os.path.join(LOG_DIR, "event_startpoints.json")

    db = load_json(OUTPUT_JSON)

    for root, _, files in os.walk(BASE_DIR):
        if "Logs" in root:
            continue
        for file in files:
            if not file.endswith("_demean_detrend_IC.mseed"):
                continue

            file_path = os.path.join(root, file)
            parts = os.path.normpath(file_path).split(os.sep)
            event_name = parts[-3] if len(parts) >= 3 else "UnknownEvent"
            station_name = parts[-2] if len(parts) >= 2 else "UnknownStation"

            try:
                st = read(file_path)
            except Exception as e:
                print(f"⚠️ Αποτυχία ανάγνωσης {file_path}: {e}")
                continue

            for tr in st:
                try:
                    from utils import aic_picker
                    idx, aic = aic_picker(tr.data.astype(float))
                    if idx is None:
                        print(f"⚠️ {event_name}/{station_name}/{tr.id}: αποτυχία AIC")
                        continue

                    t0 = tr.stats.starttime + idx / tr.stats.sampling_rate
                    aic_min = float(np.min(aic))

                    insert_result(
                        db,
                        event_name,
                        station_name,
                        tr.stats.channel,
                        int(idx),
                        t0,
                        aic_min
                    )

                    # 💾 ΑΜΕΣΗ εγγραφή στο JSON μετά από κάθε trace
                    save_json(OUTPUT_JSON, db)

                    print(f"✅ {event_name}/{station_name}/{tr.id}: start @ {t0} (sample {idx})")

                except Exception as e:
                    print(f"⚠️ AIC σφάλμα στο {event_name}/{station_name}/{tr.id}: {e}")
                    continue

    print(f"\n💾 Όλα τα αποτελέσματα έχουν αποθηκευτεί προοδευτικά στο: {OUTPUT_JSON}")

# ---------------------------------------------------------
# Εκτέλεση
# ---------------------------------------------------------
if __name__ == "__main__":
    find_event_start()
