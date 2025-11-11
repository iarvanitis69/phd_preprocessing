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
    import json
    import numpy as np
    from obspy import read
    from scipy.signal import find_peaks
    from main import LOG_DIR, BASE_DIR

    # --- Inline helper για SNR ---
    def add_min_station_snr(station_results: dict, minimum_station_snr: float):
        max_duration = 0.0
        for ch, info in station_results.items():
            if isinstance(info, dict) and "duration_time" in info:
                try:
                    dur = float(info["duration_time"])
                    if dur > max_duration:
                        max_duration = dur
                except ValueError:
                    continue
        station_results["minimum_station_snr"] = round(minimum_station_snr, 3)

    # --- Paths ---
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

    # Αν υπάρχει ήδη το PS_boundaries.json, φόρτωσέ το
    if os.path.exists(OUTPUT_JSON):
        try:
            all_results = load_json(OUTPUT_JSON)
        except Exception:
            all_results = {}
    else:
        all_results = {}

    events_dict = snr_data.get("Events", {})

    # --- Κύριος βρόχος ---
    for year, events in events_dict.items():
        for event, stations in events.items():
            for station, chans in stations.items():

                # ➤ Έλεγχος αν υπάρχει ήδη
                if (
                    str(year) in all_results
                    and event in all_results[str(year)]
                    and station in all_results[str(year)][event]
                ):
                    print(f"⏭️ Παράλειψη: {year}/{event}/{station} υπάρχει ήδη")
                    continue

                year_path = os.path.join(BASE_DIR, year)
                event_path = os.path.join(year_path, event)
                station_path = os.path.join(event_path, station)
                station_results = {}

                for root, _, files in os.walk(station_path):
                    if "info.json" in root:
                        continue
                    for fname in files:
                        # ➤ Μόνο τα Z κανάλια
                        if not fname.endswith("_demeanDetrend_IC_BPF.mseed") or "HHZ" not in fname:
                            continue
                        try:
                            st = read(os.path.join(station_path, fname))
                        except Exception as e:
                            print(f"⚠️ Αποτυχία ανάγνωσης {fname}: {e}")
                            continue

                        for tr in st:
                            try:
                                data = tr.data.astype(float)
                                sr = tr.stats.sampling_rate
                                aic_idx, _ = aic_picker(data)
                                if aic_idx is None:
                                    print(f"⚠️ AIC αποτυχία για {tr.id}")
                                    continue

                                abs_data = np.abs(data)
                                max_val = np.max(abs_data)
                                if max_val == 0:
                                    print(f"⚠️ Μηδενικό σήμα στο {tr.id}")
                                    continue
                                norm_data = abs_data / max_val

                                start_time = tr.stats.starttime + aic_idx / sr
                                buffer_samples = int(0.5 * sr)
                                search_segment = norm_data[aic_idx + buffer_samples:]
                                threshold = 0.2 * np.max(search_segment)

                                peaks, properties = find_peaks(
                                    search_segment,
                                    height=threshold,
                                    prominence=0.1,
                                    distance=int(0.3 * sr)
                                )

                                if len(peaks) == 0:
                                    pick_idx = int(aic_idx + np.argmax(search_segment))
                                else:
                                    main_peak = peaks[np.argmax(properties["peak_heights"])]
                                    pick_idx = aic_idx + buffer_samples + main_peak

                                pick_time = tr.stats.starttime + pick_idx / sr
                                pick_ampl = float(norm_data[pick_idx])
                                end_idx = 2 * pick_idx - aic_idx
                                end_time = tr.stats.starttime + end_idx / sr
                                duration_samples = int(2 * (pick_idx - aic_idx))
                                duration_time = duration_samples / sr

                                ch_id = tr.id.split('.')[-1]  # HHZ

                                station_results[ch_id] = {
                                    "start_idx": int(aic_idx),
                                    "start_time": str(start_time),
                                    "peak_amplitude_idx": int(pick_idx),
                                    "peak_amplitude_time": str(pick_time),
                                    "peak_amplitude": pick_ampl,
                                    "end_of_peak_segment_sample": int(end_idx),
                                    "end_of_peak_segment_time": str(end_time),
                                    "duration_nof_samples": duration_samples,
                                    "duration_time": str(duration_time),
                                }

                            except Exception as e:
                                print(f"⚠️ Σφάλμα στο {year}/{event}/{station}/{tr.id}: {e}")

                # ✅ Μόλις ολοκληρωθεί ο σταθμός:
                if len(station_results) > 0:
                    add_min_station_snr(station_results, chans.get("minimum_snr", 0))

                    # --- Ενημέρωση συνολικού dict με σωστή ιεραρχία ---
                    year_dict = all_results.setdefault(str(year), {})
                    event_dict = year_dict.setdefault(event, {})
                    event_dict[station] = station_results

                    # --- Εγγραφή στο αρχείο ---
                    save_json(OUTPUT_JSON, all_results)

                    print(
                        f'💾 Αποθηκεύτηκαν τα αποτελέσματα για {year}/{event}/{station}: '
                        f'minimum_station_snr={chans.get("minimum_snr", 0)}, duration_time_HHZ={duration_time:.2f}'
                    )

    print(f"\n✅ Ολοκληρώθηκε η καταγραφή όλων των σταθμών στο: {OUTPUT_JSON}")

# ==========================================================
# ✅ ΦΑΣΗ 2: Δημιουργία αποσπασμάτων με σταθερό duration
# ==========================================================
def create_peak_segmentation_files(min_snr: float, min_duration: float, max_duration: float):
    """
    Δημιουργεί PS αρχεία (κομμένα segments) από τα BPF αρχεία,
    με βάση τα κριτήρια:
      - SNR >= min_snr
      - min_duration <= duration_time <= max_duration

    Τα νέα αρχεία αποθηκεύονται μέσα στον φάκελο κάθε σταθμού,
    σε υποφάκελο τύπου:
      SNRgt{min_snr}_DurBtwn{min_duration}_{max_duration}
    """
    import os
    from obspy import read
    from main import LOG_DIR, BASE_DIR

    # --- Paths ---
    PS_JSON = os.path.join(LOG_DIR, "PS_boundaries.json")
    if not os.path.exists(PS_JSON):
        print(f"❌ Δεν βρέθηκε το {PS_JSON}")
        return

    db = load_json(PS_JSON)

    # --- Βρόχος για κάθε event/station/channel ---
    for year, events in db.items():
        for event_name, stations in events.items():
            for station_name, channels in stations.items():
                if not isinstance(channels, dict):
                    continue

                # Έλεγχος SNR
                station_snr = channels.get("minimum_station_snr", 0)
                if station_snr < min_snr:
                    continue

                # Εύρεση path σταθμού
                station_path = os.path.join(BASE_DIR, str(year), event_name, station_name)
                if not os.path.exists(station_path):
                    continue

                # Δημιουργία υποφακέλου εξόδου μέσα στο station
                output_dir = os.path.join(
                    station_path, f"SNRgt{min_snr}_DurBtwn{int(min_duration)}_{int(max_duration)}"
                )
                os.makedirs(output_dir, exist_ok=True)

                # Έλεγχος καναλιών
                for ch_name, ch_info in channels.items():
                    if not isinstance(ch_info, dict) or "duration_time" not in ch_info:
                        continue

                    try:
                        dur = float(ch_info["duration_time"])
                    except ValueError:
                        continue

                    # --- Έλεγχος ορίων διάρκειας ---
                    if not (min_duration <= dur <= max_duration):
                        continue

                    start_idx = ch_info.get("start_idx", -1)
                    if start_idx < 0:
                        continue

                    # Εντοπισμός αρχικού αρχείου
                    orig_file = os.path.join(
                        station_path,
                        f"{station_name}..{ch_name}__{event_name}_demeanDetrend_IC_BPF.mseed"
                    )

                    if not os.path.exists(orig_file):
                        print(f"⚠️ Δεν βρέθηκε το αρχείο: {orig_file}")
                        continue

                    # --- Ανάγνωση waveform και υπολογισμός samples ---
                    try:
                        st = read(orig_file)
                        sr = st[0].stats.sampling_rate
                        duration_samples = int(round(dur * sr))
                    except Exception as e:
                        print(f"⚠️ Σφάλμα ανάγνωσης {orig_file}: {e}")
                        continue

                    # --- Δημιουργία αποκομμένου αρχείου ---
                    output_path = extract_segment_from_mseed_file(
                        input_path=orig_file,
                        start_index=start_idx,
                        duration_samples=duration_samples,
                        output_dir=output_dir
                    )

                    if output_path:
                        print(
                            f"✅ Δημιουργήθηκε: {output_path} "
                            f"(SNR={station_snr:.2f}, duration={dur:.2f}s)"
                        )

    print(f"\n✅ Ολοκληρώθηκε η δημιουργία PS αρχείων για SNR ≥ {min_snr}, "
          f"διάρκεια μεταξύ {min_duration}–{max_duration} sec.")


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

def plot_station_duration_distribution(json_path: str = None, bin_size: float = 10.0):
    """
    Υπολογίζει και σχεδιάζει την κατανομή (ραβδόγραμμα)
    των duration_time τιμών ΜΟΝΟ για τα Z κανάλια (π.χ. HHZ, BHZ, EHZ)
    από το αρχείο PS_boundaries.json και το αποθηκεύει στο Logs/station-duration-distribution.png
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from main import LOG_DIR

    # --- Αν δεν δοθεί path, πάρε το προεπιλεγμένο ---
    if json_path is None:
        json_path = os.path.join(LOG_DIR, "PS_boundaries.json")

    # --- Ανάγνωση δεδομένων ---
    if not os.path.exists(json_path):
        print(f"❌ Δεν βρέθηκε το αρχείο: {json_path}")
        return

    data = load_json(json_path)
    durations = []

    # --- Βήμα 1: Συλλογή duration_time μόνο από Z κανάλια ---
    for event_name, stations in data.items():
        for station_name, channels in stations.items():
            if not isinstance(channels, dict):
                continue

            for ch_name, ch_info in channels.items():
                if not isinstance(ch_info, dict):
                    continue
                if not ch_name.endswith("Z"):  # Μόνο τα Z κανάλια (π.χ. HHZ)
                    continue

                dur = ch_info.get("duration_time")
                if dur is None:
                    continue
                try:
                    durations.append(float(dur))
                except ValueError:
                    continue

    if not durations:
        print("❌ Δεν βρέθηκαν τιμές duration_time για κανάλια Z")
        return

    # --- Βήμα 2: Δημιουργία bins ---
    max_value = max(durations)
    bins = np.arange(0, max_value + bin_size, bin_size)

    # --- Βήμα 3: Σχεδίαση ραβδογράμματος ---
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(durations, bins=bins, color="teal", edgecolor="black", alpha=0.8)

    plt.title("Κατανομή Duration (μόνο Z κανάλια)", fontsize=14, fontweight="bold")
    plt.xlabel("Διάρκεια (δευτερόλεπτα)", fontsize=12)
    plt.ylabel("Πλήθος σταθμών", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Προσθήκη labels πάνω από κάθε μπάρα
    for c, p in zip(counts, patches):
        if c > 0:
            plt.text(p.get_x() + p.get_width() / 2, c, f"{int(c)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    # --- Βήμα 4: Αποθήκευση στο Logs ---
    output_png = os.path.join(LOG_DIR, "station-duration-distribution.png")
    plt.savefig(output_png, dpi=200)
    print(f"💾 Αποθηκεύτηκε το ραβδόγραμμα στο {output_png}")

    # --- Προαιρετική εμφάνιση ---
    plt.show()

# ==========================================================
if __name__ == "__main__":
    find_peak_segmentation()
    #create_peak_segmentation_files()
