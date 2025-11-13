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
def find_boundaries():
    import os
    import json
    import numpy as np
    from obspy import read
    from scipy.signal import find_peaks, butter, filtfilt, hilbert
    from main import LOG_DIR, BASE_DIR

    # --- Helper: αποθήκευση JSON με atomic τρόπο ---
    def save_json(path, data):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)

    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --- Helper: υπολογισμός ελάχιστου SNR ---
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

    # --- Bandpass φίλτρο 1–20 Hz ---
    def bandpass_filter(data, sr, fmin=1.0, fmax=20.0, order=4):
        nyquist = 0.5 * sr
        low = fmin / nyquist
        high = fmax / nyquist
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, data)

    # --- Paths ---
    OUTPUT_JSON = os.path.join(LOG_DIR, "boundaries.json")
    AIC_FAIL_JSON = os.path.join(LOG_DIR, "AIC_failure.json")
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

    # --- Αν υπάρχει ήδη το boundaries.json, φόρτωσέ το ---
    if os.path.exists(OUTPUT_JSON):
        try:
            all_results = load_json(OUTPUT_JSON)
            print(f"📂 Φορτώθηκε το υπάρχον boundaries.json")
        except Exception:
            all_results = {}
    else:
        all_results = {}

    events_dict = snr_data.get("Events", {})

    # --- Κύριος βρόχος ---
    for year, events in events_dict.items():
        for eventJson, stations in events.items():
            for stationJson, chans in stations.items():
                # Skip αν υπάρχει ήδη
                if (
                    str(year) in all_results
                    and eventJson in all_results[str(year)]
                    and stationJson in all_results[str(year)][eventJson]
                ):
                    print(f"⏭️ Παράλειψη: {year}/{eventJson}/{stationJson} υπάρχει ήδη")
                    continue

                year_path = os.path.join(BASE_DIR, year)
                event_path = os.path.join(year_path, eventJson)
                station_path = os.path.join(event_path, stationJson)
                station_results = {}

                # SNR του σταθμού
                station_snr = chans.get("minimum_snr", 0)

                for root, _, files in os.walk(station_path):
                    if "info.json" in root:
                        continue
                    for fname in files:
                        if not fname.endswith("_demeanDetrend_IC_BPF.mseed") or "HHZ" not in fname:
                            continue
                        try:
                            st = read(os.path.join(station_path, fname))
                        except Exception as e:
                            print(f"⚠️ Αποτυχία ανάγνωσης {year}/{eventJson}/{stationJson} {fname}: {e}")
                            continue

                        channelSnrJson = chans.get("HHZ", 0).get("snr", 0)

                        for tr in st:
                            try:
                                data = tr.data.astype(float)
                                sr = tr.stats.sampling_rate

                                # --- Βήμα 1: Εύρεση AIC έναρξης ---
                                start_of_event_idx, _ = aic_picker(data)
                                if start_of_event_idx is None:
                                    # --- Καταγραφή AIC αποτυχίας (ΝΕΑ ΔΟΜΗ) ---
                                    try:
                                        aic_failures = load_json(AIC_FAIL_JSON) if os.path.exists(AIC_FAIL_JSON) else {}
                                    except Exception:
                                        aic_failures = {}

                                    # root counter
                                    current_count = aic_failures.get("count", 0)

                                    # year → event → station (no channel)
                                    year_dict = aic_failures.setdefault(str(year), {})
                                    event_dict = year_dict.setdefault(eventJson, {})

                                    # αν ο σταθμός δεν υπάρχει ήδη, πρόσθεσέ τον και αύξησε count
                                    if stationJson not in event_dict:
                                        event_dict[stationJson] = True
                                        aic_failures["count"] = current_count + 1

                                    save_json(AIC_FAIL_JSON, aic_failures)
                                    continue

                                # --- Βήμα 2: Bandpass 1–20 Hz ---
                                filtered = bandpass_filter(data, sr, 1.0, 20.0)

                                # --- Βήμα 3: Hilbert envelope ---
                                envelope = np.abs(hilbert(filtered))
                                norm_env = envelope / (np.max(envelope) or 1.0)

                                # --- Βήμα 4: Buffer 0.5 s μετά το AIC ---
                                buffer_samples = int(0.5 * sr)
                                search_segment = norm_env[start_of_event_idx + buffer_samples:]
                                threshold = 0.2 * np.max(search_segment)

                                # --- Βήμα 5: Εύρεση peaks ---
                                peaks, properties = find_peaks(
                                    search_segment,
                                    height=threshold,
                                    prominence=0.8,
                                    distance=int(0.3 * sr)
                                )
                                if len(peaks) == 0:
                                    peak_amplitude_idx = int(start_of_event_idx + np.argmax(search_segment))
                                else:
                                    peak_amplitude_idx = peaks[0]
                                    peak_amplitude_idx = peak_amplitude_idx + start_of_event_idx
                                    # peak_amplitude_idx = start_of_event_idx + buffer_samples + main_peak

                                # --- Βήμα 6: Υπολογισμός χρόνων ---
                                start_of_event_datetime = tr.stats.starttime + start_of_event_idx / sr
                                peak_amplitude_datetime = tr.stats.starttime + peak_amplitude_idx / sr
                                end_of_peak_segment_idx = 2 * peak_amplitude_idx - start_of_event_idx
                                end_of_peak_segment_datetime = tr.stats.starttime + end_of_peak_segment_idx / sr
                                peak_segment_duration_samples = int(2 * (peak_amplitude_idx - start_of_event_idx))
                                peak_segment_duration_time = peak_segment_duration_samples / sr

                                pick_ampl = float(norm_env[peak_amplitude_idx])
                                ch_id = tr.id.split('.')[-1]

                                # --- Βήμα 7: Υπολογισμός τέλους σήματος με βάση το SNR ---
                                threshold_end = 1.0 / (channelSnrJson or 1.0)
                                end_of_event_idx = None
                                for i in range(peak_amplitude_idx, len(norm_env)):
                                    if norm_env[i] <= threshold_end:
                                        end_of_event_idx = i
                                        break
                                if end_of_event_idx is None:
                                    end_of_event_idx = len(norm_env) - 1

                                end_of_event_idx = end_of_event_idx + start_of_event_idx
                                end_of_event_time = tr.stats.starttime + end_of_event_idx / sr

                                # --- Υπολογισμός συνολικής διάρκειας ---
                                clean_event_duration_nof_samples = int(end_of_event_idx) - int(start_of_event_idx)
                                clean_event_duration_time = clean_event_duration_nof_samples / sr
                                total_signal_nof_samples = len(tr.data)
                                total_signal_time = total_signal_nof_samples / sr

                                # --- Αποθήκευση ---
                                station_results[ch_id] = {
                                    "start_of_event_idx": int(start_of_event_idx),
                                    "start_of_event_datetime": str(start_of_event_datetime),
                                    "peak_amplitude_idx": int(peak_amplitude_idx),
                                    "peak_amplitude_datetime": str(peak_amplitude_datetime),
                                    "peak_amplitude": round(pick_ampl, 5),
                                    "end_of_peak_segment_idx": int(end_of_peak_segment_idx),
                                    "end_of_peak_segment_datetime": str(end_of_peak_segment_datetime),
                                    "peak_segment_duration_nof_samples": int(peak_segment_duration_samples),
                                    "peak_segment_duration_time": f"{peak_segment_duration_time:.2f}",
                                    "end_of_event_idx": int(end_of_event_idx),
                                    "end_of_event_time": str(end_of_event_time),
                                    "clean_event_duration_nof_samples": int(clean_event_duration_nof_samples),
                                    "clean_event_duration_time": f"{clean_event_duration_time:.2f}",
                                    "total_signal_nof_samples": int(total_signal_nof_samples),
                                    "total_signal_time": f"{total_signal_time:.2f}",
                                }

                            except Exception as e:
                                print(f"⚠️ Σφάλμα στο {year}/{eventJson}/{stationJson}/{tr.id}: {e}")

                # ✅ Ο σταθμός ολοκληρώθηκε
                if len(station_results) > 0:
                    add_min_station_snr(station_results, station_snr)

                    # --- Μετρητής σταθμών ---
                    total_key = "total_nof_stations"
                    prev = all_results.get(total_key, 0)
                    all_results[total_key] = prev + 1

                    # --- Ενημέρωση δομής ---
                    year_dict = all_results.setdefault(str(year), {})
                    event_dict = year_dict.setdefault(eventJson, {})
                    event_dict[stationJson] = station_results

                    save_json(OUTPUT_JSON, all_results)

                    print(
                        f"💾 Αποθηκεύτηκε {year}/{eventJson}/{stationJson}: "
                        f"SNR={station_snr:.2f}, peak_segment_duration_time={peak_segment_duration_time:.2f}, "
                        f"clean_event_duration_time={clean_event_duration_time:.2f}, total_signal_time={total_signal_time:.2f}"
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
    PS_JSON = os.path.join(LOG_DIR, "boundaries.json")
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
                            f"(SNR={float(station_snr):.2f}, duration={dur:.2f}s)"
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


def plot_peak_segmentation_duration_distribution(bin_size: float = 5.0):
    """
    Υπολογίζει και σχεδιάζει την κατανομή (ραβδόγραμμα)
    των duration_of_peak_segment τιμών ΜΟΝΟ για τα Z κανάλια (π.χ. HHZ, BHZ, EHZ)
    από το αρχείο boundaries.json και το αποθηκεύει στο Logs/station-duration-distribution.png
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from main import LOG_DIR

    # --- Ανάγνωση αρχείου ---
    json_path = os.path.join(LOG_DIR, "PS_boundaries.json")
    if not os.path.exists(json_path):
        print(f"❌ Δεν βρέθηκε το αρχείο: {json_path}")
        return

    data = load_json(json_path)
    durations = []

    # --- Διασχίζουμε τη δομή: έτος → event → σταθμό → κανάλι ---
    for year, events in data.items():
        if year == "total_nof_stations":
            continue
        for event_name, stations in events.items():
            for station_name, channels in stations.items():
                if not isinstance(channels, dict):
                    continue

                # Μόνο τα κανάλια Z (HHZ, BHZ, EHZ)
                for ch_name, ch_info in channels.items():
                    if not isinstance(ch_info, dict):
                        continue
                    if not ch_name.endswith("Z"):
                        continue

                    dur = ch_info.get("peak_segment_duration_time")
                    if dur is None:
                        continue

                    try:
                        durations.append(float(dur))
                    except ValueError:
                        continue

    if not durations:
        print("❌ Δεν βρέθηκαν τιμές duration_of_peak_segment για κανάλια Z")
        return

    # --- Bins ---
    max_value = max(durations)
    bins = np.arange(0, max_value + bin_size, bin_size)

    # --- Ραβδόγραμμα ---
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(durations, bins=bins, color="teal", edgecolor="black", alpha=0.8)

    plt.title("Distribution Peak Segmentation Duration (only Z channels)", fontsize=14, fontweight="bold")
    plt.xlabel("Duration(sec)", fontsize=12)
    plt.ylabel("Nof Stations", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Προσθήκη labels πάνω από κάθε μπάρα
    for c, p in zip(counts, patches):
        if c > 0:
            plt.text(p.get_x() + p.get_width() / 2, c, f"{int(c)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    # --- Αποθήκευση ---
    output_png = os.path.join(LOG_DIR, "station-duration-distribution.png")
    plt.savefig(output_png, dpi=200)
    print(f"💾 Histogram stored at {output_png}")

    plt.show()

def plot_clean_event_duration_distribution(bin_size: float = 5.0):
    """
    Υπολογίζει και σχεδιάζει την κατανομή (ραβδόγραμμα)
    των event_duration_time τιμών ΜΟΝΟ για τα Z κανάλια (HHZ, BHZ, EHZ)
    από το αρχείο boundaries.json και το αποθηκεύει στο Logs/clean-event-duration-distribution.png
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from main import LOG_DIR

    # --- Ανάγνωση αρχείου ---
    json_path = os.path.join(LOG_DIR, "PS_boundaries.json")
    if not os.path.exists(json_path):
        print(f"❌ Δεν βρέθηκε το αρχείο: {json_path}")
        return

    data = load_json(json_path)
    durations = []

    # --- Διασχίζουμε τη δομή: year → event → station → channel ---
    for year, events in data.items():
        if year == "total_nof_stations":
            continue  # skip global key

        for event_name, stations in events.items():
            for station_name, channels in stations.items():
                if not isinstance(channels, dict):
                    continue

                # μόνο τα Z κανάλια
                for ch_name, ch_info in channels.items():
                    if not isinstance(ch_info, dict):
                        continue
                    if not ch_name.endswith("Z"):
                        continue

                    dur = ch_info.get("clean_event_duration_time")
                    if dur is None:
                        continue

                    try:
                        durations.append(float(dur))
                    except ValueError:
                        continue

    if not durations:
        print("❌ Δεν βρέθηκαν τιμές event_duration_time για κανάλια Z")
        return

    # --- Bins ---
    max_value = max(durations)
    bins = np.arange(0, max_value + bin_size, bin_size)

    # --- Ραβδόγραμμα ---
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(durations, bins=bins, color="purple", edgecolor="black", alpha=0.8)

    plt.title("Distribution of Clean Event Duration (Z channels only)", fontsize=14, fontweight="bold")
    plt.xlabel("Duration (seconds)", fontsize=12)
    plt.ylabel("Number of stations", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # αριθμοί πάνω από κάθε μπάρα
    for c, p in zip(counts, patches):
        if c > 0:
            plt.text(p.get_x() + p.get_width() / 2, c, f"{int(c)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    # --- Αποθήκευση ---
    output_png = os.path.join(LOG_DIR, "clean-event-duration-distribution.png")
    plt.savefig(output_png, dpi=200)
    print(f"💾 Αποθηκεύτηκε στο {output_png}")

    plt.show()

def plot_snr_distribution(bin_size: float = 3.0):
    """
    Υπολογίζει και σχεδιάζει την κατανομή (ραβδόγραμμα)
    των minimum_station_snr τιμών από το αρχείο boundaries.json
    και το αποθηκεύει στο Logs/snr-distribution.png
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from main import LOG_DIR

    # --- Διαδρομή αρχείου ---
    json_path = os.path.join(LOG_DIR, "PS_boundaries.json")

    # --- Έλεγχος ύπαρξης ---
    if not os.path.exists(json_path):
        print(f"❌ Δεν βρέθηκε το αρχείο: {json_path}")
        return

    # --- Ανάγνωση δεδομένων ---
    data = load_json(json_path)
    snr_values = []

    # --- Δομή: έτος → γεγονός → σταθμός ---
    for year, events in data.items():
        if not isinstance(events, dict):
            continue
        for event_name, stations in events.items():
            if not isinstance(stations, dict):
                continue
            for station_name, station_info in stations.items():
                if not isinstance(station_info, dict):
                    continue

                # Αν υπάρχει τιμή minimum_station_snr στο επίπεδο σταθμού
                min_snr = station_info.get("minimum_station_snr")
                if min_snr is None:
                    continue

                try:
                    snr_values.append(float(min_snr))
                except (TypeError, ValueError):
                    continue

    if not snr_values:
        print("❌ Δεν βρέθηκαν τιμές minimum_station_snr στο boundaries.json")
        return

    # --- Δημιουργία bins ---
    max_value = max(snr_values)
    bins = np.arange(0, max_value + bin_size, bin_size)

    # --- Ραβδόγραμμα ---
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(
        snr_values, bins=bins, color="orange", edgecolor="black", alpha=0.8
    )

    plt.title("Distribution SNR per station", fontsize=14, fontweight="bold")
    plt.xlabel("SNR (value per station)", fontsize=12)
    plt.ylabel("Nof Stations", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Προσθήκη labels πάνω από κάθε μπάρα
    for c, p in zip(counts, patches):
        if c > 0:
            plt.text(
                p.get_x() + p.get_width() / 2,
                c,
                f"{int(c)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    # --- Αποθήκευση ---
    output_png = os.path.join(LOG_DIR, "snr-distribution.png")
    plt.savefig(output_png, dpi=200)
    print(f"💾 Histogram stored at {output_png}")

    plt.show()

def count_nof_training_stations(
        min_snr: float,
        max_ps_duration: float,
        min_clean_event_duration: float):
    """
    Classifies Z-channel signals from PS_boundaries.json into:

    A) Training-eligible:
         - minimum_station_snr >= min_snr
         - peak_segment_duration_time <= max_ps_duration
         - clean_event_duration_time >= min_clean_event_duration

    B1) High SNR (>= min_snr) but TOO LONG peak segment (> max_ps_duration s)
    B2) Low SNR (< min_snr)
    B3) High SNR (>= min_snr) but TOO SHORT clean event (< min_clean_event_duration s)
    """

    import os
    from main import LOG_DIR

    json_path = os.path.join(LOG_DIR, "PS_boundaries.json")

    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return

    data = load_json(json_path)

    # --- Counters ---
    to_training = 0
    high_snr_and_high_ps_duration = 0
    low_snr = 0
    high_snr_but_low_clean_event = 0

    # --- Traverse structure: year → event → station → channel ---
    for year, events in data.items():
        if year == "total_nof_stations":
            continue
        if not isinstance(events, dict):
            continue

        for event_name, stations in events.items():
            for station_name, channels in stations.items():

                # Station SNR
                station_snr = channels.get("minimum_station_snr")
                if station_snr is None:
                    continue
                station_snr = float(station_snr)

                # --- For every Z channel ---
                for ch_name, ch_info in channels.items():
                    if not isinstance(ch_info, dict):
                        continue
                    if not ch_name.endswith("Z"):
                        continue

                    # --- Peak Segmentation Duration ---
                    ps_dur = ch_info.get("peak_segment_duration_time")
                    if ps_dur is None:
                        continue
                    try:
                        ps_dur = float(ps_dur)
                    except:
                        continue

                    # --- Clean Event Duration ---
                    clean_dur = ch_info.get("clean_event_duration_time")
                    if clean_dur is None:
                        continue
                    try:
                        clean_dur = float(clean_dur)
                    except:
                        continue

                    # ------------------------------------------------------
                    # CATEGORY B2 — LOW SNR (< min_snr)
                    # ------------------------------------------------------
                    if station_snr < min_snr:
                        low_snr += 1
                        continue

                    # ------------------------------------------------------
                    # CATEGORY B1 — HIGH SNR (>= min_snr) BUT TOO LONG PS duration (> max_ps_duration)
                    # ------------------------------------------------------
                    if ps_dur > max_ps_duration:
                        high_snr_and_high_ps_duration += 1
                        continue

                    # ------------------------------------------------------
                    # CATEGORY B3 — HIGH SNR (>= min_snr) BUT CLEAN EVENT TOO SHORT (< min_clean_event_duration)
                    # ------------------------------------------------------
                    if clean_dur < min_clean_event_duration:
                        high_snr_but_low_clean_event += 1
                        continue

                    # ------------------------------------------------------
                    # CATEGORY A — Training-eligible signals
                    # ------------------------------------------------------
                    if (
                        station_snr >= min_snr and
                        ps_dur <= max_ps_duration and
                        clean_dur >= min_clean_event_duration
                    ):
                        to_training += 1

    # --- PRINT REPORT ---
    print("\n📊 *** SIGNAL CLASSIFICATION REPORT ***")
    print(f"⚠ NOT USED SET : SNR ≥ {min_snr} & PS_duration_time > {max_ps_duration} sec : {high_snr_and_high_ps_duration}")
    print(f"⚠ NOT USED SET : SNR < {min_snr} : {low_snr}")
    print(f"⚠ NOT USED SET : SNR ≥ {min_snr} & clean_event_duration_time < {min_clean_event_duration} sec : {high_snr_but_low_clean_event}")
    print("------------------------------------------------------------------------------------------------------------")
    print(f"✔ TRAINING SET : SNR ≥ {min_snr} & PS_duration_time ≤ {max_ps_duration} sec "
          f"& clean_event_duration_time ≥ {min_clean_event_duration} sec : {to_training}")

# ==========================================================
if __name__ == "__main__":
    #find_boundaries()
    #plot_clean_event_duration_distribution()
    #plot_peak_segmentation_duration_distribution()
    #plot_snr_distribution()
    count_nof_training_stations(5,30,30)
    #create_peak_segmentation_files()
