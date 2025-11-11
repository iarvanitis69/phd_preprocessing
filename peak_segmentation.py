#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from obspy import read, Trace, Stream

#from main import LOG_DIR

#from scipy.stats import pairs



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


def insert_channel_result(db_path, event, station, channel, result):
    db = load_json(db_path)

    ev = db.setdefault(event, {})
    st = ev.setdefault(station, {})
    st[channel] = result

    save_json(db_path, db)

def add_station_max_duration_and_min_station_snr(json_path: str, event_name: str, station_name: str, minimun_station_snr:float):
    """
    Διαβάζει ένα JSON αρχείο (π.χ. PS_boundaries.json),
    εντοπίζει τον συγκεκριμένο σταθμό, υπολογίζει το μέγιστο duration_time
    και προσθέτει το κλειδί 'Max Station Duration' στο ίδιο επίπεδο.

    :param json_path: Πλήρες path προς το αρχείο JSON
    :param event_name: Το όνομα του event (π.χ. '20250209T152330_36.60_25.66_7.0km_M3.2')
    :param station_name: Το όνομα του σταθμού (π.χ. 'HL.AMGA')
    """
    data = load_json(json_path)

    # Έλεγχος ότι υπάρχει το event και ο σταθμός
    event_dict = data.get(event_name)
    if event_dict is None:
        print(f"❌ Δεν βρέθηκε το event: {event_name}")
        return

    station_dict = event_dict.get(station_name)
    if station_dict is None:
        print(f"❌ Δεν βρέθηκε ο σταθμός: {station_name}")
        return

    # Υπολογισμός μέγιστης διάρκειας από τα κανάλια
    max_duration = 0.0
    for channel, info in station_dict.items():
        if isinstance(info, dict) and "duration_time" in info:
            try:
                dur = float(info["duration_time"])
                if dur > max_duration:
                    max_duration = dur
            except ValueError:
                continue

    # Προσθήκη του νέου πεδίου
    station_dict["max_station_duration"] = round(max_duration, 3)
    station_dict["minimun_station_snr"] = round(minimun_station_snr, 3)

    # Ενημέρωση του JSON
    event_dict[station_name] = station_dict
    data[event_name] = event_dict
    save_json(json_path, data)

    print(f"✅ Προστέθηκε 'στο {station_name} ({event_name}) Max Station Duration': {max_duration:.3f}")


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
        output_filename = f"{base}_PS.mseed"
        output_path = os.path.join(folder, output_filename)

        trimmed.write(output_path, format="MSEED")
        return output_path

    except Exception as e:
        print(f"❌ Σφάλμα στο extract_segment_from_mseed_file: {e}")
        return None

from typing import List, Set, Tuple

# ==========================================================
# ✅ ΦΑΣΗ 1: Υπολογισμός start/pick/end & ενημέρωση JSON
# ==========================================================
def find_peak_segmentation():
    import os
    import numpy as np
    from obspy import read
    from scipy.signal import find_peaks
    from main import LOG_DIR, BASE_DIR

    OUTPUT_JSON = os.path.join(LOG_DIR, "PS_boundaries.json")
    os.makedirs(LOG_DIR, exist_ok=True)

    snr_path = os.path.join(LOG_DIR, "snr.json")
    if not os.path.exists(snr_path):
        print(f"❌ Δεν βρέθηκε το {snr_path}")
        return set()

    try:
        snr_data = load_json(snr_path)
    except Exception as e:
        print(f"⚠️ Σφάλμα κατά την ανάγνωση του snr.json: {e}")
        return set()

    events_dict = snr_data.get("Events", {})

    for year, events in events_dict.items():
        for event, stations in events.items():
            for station, chans in stations.items():
                year_path = os.path.join(BASE_DIR, year)
                event_path = os.path.join(year_path, event)
                station_path = os.path.join(event_path, station)

                for root, _, channels in os.walk(station_path):
                    if "info.json" in root:
                        continue
                    for channel in channels:
                        if not channel.endswith("_demeanDetrend_IC_BPF.mseed"):
                            continue
                        try:
                            st = read(os.path.join(station_path, channel))
                        except Exception as e:
                            print(f"⚠️ Αποτυχία ανάγνωσης {channel}: {e}")
                            continue

                        for tr in st:
                            try:
                                data = tr.data.astype(float)
                                sr = tr.stats.sampling_rate
                                aic_idx, aic_curve = aic_picker(data)
                                if aic_idx is None:
                                    print(f"⚠️ AIC αποτυχία για {tr.id}")
                                    continue

                                # --- Κανονικοποίηση ---
                                abs_data = np.abs(data)
                                max_val = np.max(abs_data)
                                if max_val == 0:
                                    print(f"⚠️ Μηδενικό σήμα στο {tr.id}")
                                    continue
                                norm_data = abs_data / max_val

                                # --- Εύρεση αρχής ---
                                start_time = tr.stats.starttime + aic_idx / sr

                                # --- Αναζήτηση peaks μετά το AIC ---
                                search_segment = norm_data[aic_idx + 1:]
                                mean_val = np.mean(search_segment)

                                # Εφάρμοσε buffer 0.5s μετά το AIC για να αποφύγεις θόρυβο
                                buffer_samples = int(0.5 * sr)
                                search_segment = norm_data[aic_idx + buffer_samples:]

                                # Ορισμός threshold πιο αυστηρού (π.χ. 20% της μέγιστης τιμής)
                                threshold = 0.2 * np.max(search_segment)

                                # Εύρεση peaks με ελάχιστη απόσταση 0.3s μεταξύ τους
                                peaks, properties = find_peaks(
                                    search_segment,
                                    height=threshold,
                                    prominence=0.1,
                                    distance=int(0.3 * sr)
                                )

                                if len(peaks) == 0:
                                    pick_idx = int(aic_idx + np.argmax(search_segment))
                                else:
                                    # Διάλεξε το πιο υψηλό peak, όχι το πρώτο
                                    main_peak = peaks[np.argmax(properties["peak_heights"])]
                                    pick_idx = aic_idx + buffer_samples + main_peak

                                pick_time = tr.stats.starttime + pick_idx / sr
                                pick_ampl = float(norm_data[pick_idx])  # normalized amplitude

                                # --- Ορισμός τέλους τμήματος ---
                                end_idx = 2 * pick_idx - aic_idx
                                end_time = tr.stats.starttime + end_idx / sr
                                duration_samples = int(2 * (pick_idx - aic_idx))
                                duration_time = duration_samples / sr

                                ch_id = tr.id.split('.')[-1]

                                insert_channel_result(
                                    OUTPUT_JSON,
                                    os.path.basename(event_path),
                                    os.path.basename(station_path),
                                    ch_id,
                                    {
                                        "start_idx": int(aic_idx),
                                        "start_time": str(start_time),
                                        "peak_amplitude_idx": int(pick_idx),
                                        "peak_amplitude_time": str(pick_time),
                                        "peak_amplitude": pick_ampl,
                                        "end_of_peak_segment_sample": int(end_idx),
                                        "end_of_peak_segment_time": str(end_time),
                                        "duration_nof_samples": duration_samples,
                                        "duration_time": str(duration_time),
                                    },
                                )

                                print(f"✅ {os.path.basename(event_path)}/{os.path.basename(station_path)}/{tr.id}: {start_time} → {end_time}")

                            except Exception as e:
                                print(f"⚠️ Σφάλμα στο {os.path.basename(event_path)}/{os.path.basename(station_path)}/{tr.id}: {e}")

                # --- Ενημέρωση SNR και διάρκειας ---
                add_station_max_duration_and_min_station_snr(
                    OUTPUT_JSON,
                    os.path.basename(event_path),
                    os.path.basename(station_path),
                    chans.get("minimum_snr", 0),
                )

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
            if not file.endswith("_demeanDetrend_IC_BPF_PS.mseed"):
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

def aic_picker(trace_data):
    """
    Υπολογίζει το AIC σε ολόκληρο το σήμα (μέχρι το pick) και επιστρέφει
    το index όπου ελαχιστοποιείται, ως πιθανή έναρξη του σεισμικού κύματος.

    :param trace_data: numpy array με το σεισμικό σήμα (float, demeaned)
    :return: (index_έναρξης, καμπύλη_AIC)
    """
    data = trace_data.astype(float)
    n = len(data)
    if n < 3:
        return None, np.array([])

    pick_idx = int(np.argmax(np.abs(data)))  # μέγιστη απόλυτη τιμή
    if pick_idx < 10:
        return None, np.array([])  # πολύ μικρό σήμα

    aic = np.zeros(pick_idx)

    for k in range(1, pick_idx - 1):
        var1 = np.var(data[:k]) or 1e-10
        var2 = np.var(data[k:pick_idx]) or 1e-10
        aic[k] = k * np.log(var1) + (pick_idx - k - 1) * np.log(var2)

    min_idx = int(np.argmin(aic[1:pick_idx - 1])) + 1
    return min_idx, aic

import matplotlib.pyplot as plt

def plot_station_duration_distribution(bin_size: float = 10.0, output_png: str = None):
    """
    Υπολογίζει και σχεδιάζει την κατανομή (ραβδόγραμμα)
    των Max Station Duration τιμών από το JSON.

    :param json_path: Πλήρες path προς το PS_boundaries.json
    :param bin_size: Εύρος bin (σε δευτερόλεπτα)
    :param output_png: Αν δοθεί path, σώζει το διάγραμμα σε PNG
    """
    data = load_json(OUTPUT_JSON)
    durations = []

    # --- Βήμα 1: Συλλογή όλων των max_station_duration ---
    for event_name, stations in data.items():
        for station_name, channels in stations.items():
            if not isinstance(channels, dict):
                continue
            max_dur = channels.get("max_station_duration")
            if max_dur is None:
                continue
            try:
                durations.append(float(max_dur))
            except ValueError:
                continue

    if not durations:
        print("❌ Δεν βρέθηκαν τιμές max_station_duration")
        return

    # --- Βήμα 2: Δημιουργία bins ---
    max_value = max(durations)
    bins = np.arange(0, max_value + bin_size, bin_size)

    # --- Βήμα 3: Σχεδίαση ραβδογράμματος ---
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)

    plt.title("Κατανομή Max Station Duration", fontsize=14, fontweight="bold")
    plt.xlabel("Διάρκεια σταθμού (δευτερόλεπτα)", fontsize=12)
    plt.ylabel("Πλήθος σταθμών", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Εμφάνιση των labels πάνω από τις μπάρες
    counts, _, patches = plt.hist(durations, bins=bins)
    for c, p in zip(counts, patches):
        if c > 0:
            plt.text(p.get_x() + p.get_width()/2, c, f"{int(c)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    # --- Βήμα 4: Αποθήκευση ή εμφάνιση ---
    if output_png:
        plt.savefig(output_png, dpi=200)
        print(f"💾 Αποθηκεύτηκε το ραβδόγραμμα στο {output_png}")
    else:
        plt.show()

# ==========================================================
if __name__ == "__main__":
    find_peak_segmentation()
    #create_peak_segmentation_files()
