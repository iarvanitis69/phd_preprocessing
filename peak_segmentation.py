#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from obspy import read, UTCDateTime, Trace, Stream


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


def insert_result(db, event, station, channel, result):
    ev = db.setdefault(event, {})
    st = ev.setdefault(station, {})
    st[channel] = result


def find_start_end_and_peak_of_signal_and_create_signal():
    from main import LOG_DIR, BASE_DIR
    from utils import aic_picker

    OUTPUT_JSON = os.path.join(LOG_DIR, "event_boundaries.json")
    SNR_JSON = os.path.join(LOG_DIR, "snr.json")

    db = load_json(OUTPUT_JSON)
    snr_data = load_json(SNR_JSON)

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
                    data = tr.data.astype(float)
                    sr = tr.stats.sampling_rate
                    aic_idx, aic_curve = aic_picker(data)
                    if aic_idx is None:
                        print(f"⚠️ AIC αποτυχία για {tr.id}")
                        continue

                    # Χρονική στιγμή έναρξης
                    start_time = tr.stats.starttime + aic_idx / sr
                    aic_min = float(np.min(aic_curve))

                    # Χρονική στιγμή και τιμή pick (μέγιστη απολ. τιμή)
                    abs_data = np.abs(data)
                    pick_idx = int(np.argmax(abs_data))
                    pick_ampl = float(abs_data[pick_idx])
                    pick_time = tr.stats.starttime + pick_idx / sr
                    end_of_peak_segment_idx = 2*pick_idx - aic_idx
                    end_of_peak_segment_time = tr.stats.starttime + end_of_peak_segment_idx / sr

                    ch_id = tr.id.split('.')[-1]

                    # Αποθήκευση στο JSON
                    insert_result(db, event_name, station_name, ch_id, {
                        "start_sample": int(aic_idx),
                        "start_time": str(start_time),
                        "pick_sample": int(pick_idx),
                        "pick_time": str(pick_time),
                        "pick_amplitude": pick_ampl,
                        "end_of_peak_segment_sample": int(end_of_peak_segment_idx),
                        "end_of_peak_segment_time": str(end_of_peak_segment_time)
                    })

                    save_json(OUTPUT_JSON, db)

                    # Δημιουργία αρχείου PeakSegmentation
                    seg_data = data[int(aic_idx):int(end_of_peak_segment_idx)]

                    seg_trace = Trace(data=seg_data, header=tr.stats.copy())
                    seg_trace.stats.starttime = tr.stats.starttime + aic_idx / sr

                    new_filename = file.replace(
                        "_demean_detrend_IC.mseed",
                        "_demean_detrend_IC_PeakSegmentation.mseed"
                    )
                    new_path = os.path.join(root, new_filename)

                    Stream([seg_trace]).write(new_path, format="MSEED")
                    print(f"📤 Δημιουργήθηκε: {new_path}")

                    print(f"✅ {event_name}/{station_name}/{tr.id}: {str(start_time)} → {str(end_of_peak_segment_time)}")

                except Exception as e:
                    print(f"⚠️ Σφάλμα στο {event_name}/{station_name}/{tr.id}: {e}")

    print(f"\n💾 Τα αποτελέσματα αποθηκεύτηκαν στο: {OUTPUT_JSON}")


if __name__ == "__main__":
    find_start_end_and_peak_of_signal_and_create_signal()
