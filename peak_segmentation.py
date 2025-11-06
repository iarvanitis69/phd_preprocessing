#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from obspy import read, Trace, Stream


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


def insert_result_and_update_max(db_path, event, station, channel, result):
    db = load_json(db_path)

    duration = result.get("duration_time", 0.0)
    duration = float(duration)

    current_max = float(db.get("maximum_duration_time", 0.0))
    if duration > current_max:
        db["maximum_duration_time"] = duration

    ev = db.setdefault(event, {})
    st = ev.setdefault(station, {})
    st[channel] = result

    save_json(db_path, db)


def extract_segment_from_mseed_file(input_path: str, start_index: int, duration_samples: int):
    try:
        st = read(input_path)
        trimmed = st.copy().clear()

        for tr in st:
            end_index = start_index + duration_samples
            if end_index > len(tr.data):
                print(f"⚠️ Περιορισμός end_index στο μήκος του trace ({len(tr.data)})")
                end_index = len(tr.data)

            segment = tr.data[start_index:end_index].astype(np.float32)
            if not np.all(np.isfinite(segment)):
                print(f"⚠️ Μη έγκυρες τιμές (NaN/Inf) στο {tr.id}")
                return None

            segment = np.clip(segment, -1e12, 1e12)
            seg_trace = tr.copy()
            seg_trace.data = segment
            seg_trace.stats.npts = len(segment)
            seg_trace.stats.starttime += start_index / seg_trace.stats.sampling_rate
            trimmed += seg_trace

        folder = os.path.dirname(input_path)
        base = os.path.basename(input_path).replace(".mseed", "")
        output_filename = f"{base}_PeakSegmentation.mseed"
        output_path = os.path.join(folder, output_filename)

        trimmed.write(output_path, format="MSEED")
        return output_path

    except Exception as e:
        print(f"❌ Σφάλμα στο extract_segment_from_mseed_file: {e}")
        return None


# ==========================================================
# ✅ ΦΑΣΗ 1: Υπολογισμός start/pick/end & ενημέρωση JSON
# ==========================================================
def find_start_end_and_peak_of_signal():
    from main import LOG_DIR, BASE_DIR
    from utils import aic_picker

    OUTPUT_JSON = os.path.join(LOG_DIR, "event_boundaries.json")
    os.makedirs(LOG_DIR, exist_ok=True)  # Ensure Logs/ exists

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

                    start_time = tr.stats.starttime + aic_idx / sr
                    abs_data = np.abs(data)
                    pick_idx = int(np.argmax(abs_data))
                    pick_time = tr.stats.starttime + pick_idx / sr
                    pick_ampl = float(abs_data[pick_idx])

                    end_idx = 2 * pick_idx - aic_idx
                    end_time = tr.stats.starttime + end_idx / sr
                    duration_samples = int(2 * (pick_idx - aic_idx))
                    duration_time = end_time - start_time

                    ch_id = tr.id.split('.')[-1]

                    insert_result_and_update_max(OUTPUT_JSON, event_name, station_name, ch_id, {
                        "start_idx": int(aic_idx),
                        "start_time": str(start_time),
                        "peak_amplitude_idx": int(pick_idx),
                        "peak_amplitude_time": str(pick_time),
                        "peak_amplitude": pick_ampl,
                        "end_of_peak_segment_sample": int(end_idx),
                        "end_of_peak_segment_time": str(end_time),
                        "duration_nof_samples": duration_samples,
                        "duration_time": str(duration_time)
                    })

                    print(f"✅ {event_name}/{station_name}/{tr.id}: {str(start_time)} → {str(end_time)}")

                except Exception as e:
                    print(f"⚠️ Σφάλμα στο {event_name}/{station_name}/{tr.id}: {e}")

    print(f"\n💾 Αποθηκεύτηκαν στο: {OUTPUT_JSON}")


# ==========================================================
# ✅ ΦΑΣΗ 2: Δημιουργία αποσπασμάτων με σταθερό duration
# ==========================================================
def create_peak_segmentation_files():
    from main import LOG_DIR, BASE_DIR
    OUTPUT_JSON = os.path.join(LOG_DIR, "event_boundaries.json")
    db = load_json(OUTPUT_JSON)

    max_duration = float(db.get("maximum_duration_time", 0.0))
    if max_duration <= 0:
        print("❌ Δεν βρέθηκε έγκυρο μέγιστο duration.")
        return

    duration_samples = None

    for root, _, files in os.walk(BASE_DIR):
        if "Logs" in root:
            continue
        for file in files:
            if not file.endswith("_demean_detrend_IC.mseed"):
                continue

            file_path = os.path.join(root, file)
            try:
                st = read(file_path)
            except Exception as e:
                print(f"⚠️ Σφάλμα ανάγνωσης {file_path}: {e}")
                continue

            for tr in st:
                event_name = os.path.normpath(file_path).split(os.sep)[-3]
                station_name = os.path.normpath(file_path).split(os.sep)[-2]
                ch_id = tr.id.split('.')[-1]

                start_idx = int(db.get(event_name, {})
                                   .get(station_name, {})
                                   .get(ch_id, {})
                                   .get("start_idx", -1))

                if start_idx < 0:
                    continue

                sr = tr.stats.sampling_rate
                if duration_samples is None:
                    duration_samples = int(round(max_duration * sr))

                output_path = extract_segment_from_mseed_file(
                    input_path=file_path,
                    start_index=start_idx,
                    duration_samples=duration_samples
                )

                if output_path:
                    print(f"✅ Δημιουργήθηκε: {output_path}")


# ==========================================================
if __name__ == "__main__":
    find_start_end_and_peak_of_signal()
    #create_peak_segmentation_files()
