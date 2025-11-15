#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from obspy import read, Trace, Stream
# from scipy.stats import deprmsg


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
def create_cutted_signal_files(min_snr: float, max_ps_duration: float):
    """
    Δημιουργεί PS αρχεία (peak-segmented MiniSEED) από τα BPF αρχεία.

    Επιλέγει μόνο:
        - minimum_station_snr >= min_snr
        - peak_segment_duration_time <= max_ps_duration

    Το νέο αρχείο κόβεται από:
        start_of_event_idx  -->  end_of_peak_segment_idx

    Και αποθηκεύεται ως:
        <station>..<channel>__<event>_demeanDetrend_IC_BPF__PS__duration_LE_<max_ps_duration>__SNR_GE_<min_snr>.mseed
    """

    import os
    from obspy import read
    from main import LOG_DIR, BASE_DIR

    PS_JSON = os.path.join(LOG_DIR, "boundaries.json")
    if not os.path.exists(PS_JSON):
        print(f"❌ Δεν βρέθηκε το {PS_JSON}")
        return

    db = load_json(PS_JSON)

    # --------------------------------------------------------------------
    # Loop σε όλα τα events / stations / channels
    # --------------------------------------------------------------------
    for year, events in db.items():
        if year == "total_nof_stations":
            continue

        for event_name, stations in events.items():
            for station_name, channels in stations.items():

                # --- Station-level SNR ---
                station_snr = channels.get("minimum_station_snr", 0)
                if station_snr < min_snr:
                    continue

                # Path του station
                station_path = os.path.join(BASE_DIR, str(year), event_name, station_name)
                if not os.path.exists(station_path):
                    print(f"⚠️ Δεν υπάρχει ο φάκελος: {station_path}")
                    continue

                # Output subfolder
                output_dir = os.path.join(
                    station_path,
                    f"PS_SNR_GE_{min_snr}__DUR_LE_{max_ps_duration}"
                )
                os.makedirs(output_dir, exist_ok=True)

                # ----------------------------------------------------------------
                # Loop σε channels
                # ----------------------------------------------------------------
                for ch_name, ch_info in channels.items():
                    if not isinstance(ch_info, dict):
                        continue
                    if not ch_name.endswith("Z"):
                        continue

                    # Duration filter
                    ps_duration = ch_info.get("peak_segment_duration_time")
                    if ps_duration is None:
                        continue

                    try:
                        ps_duration = float(ps_duration)
                    except:
                        continue

                    if ps_duration > max_ps_duration:
                        continue

                    # Required boundaries
                    start_idx = ch_info.get("start_of_event_idx")
                    end_peak_idx = ch_info.get("end_of_peak_segment_idx")

                    if start_idx is None or end_peak_idx is None:
                        continue

                    # ----------------------------------------------------------------
                    # Original file path
                    # ----------------------------------------------------------------
                    orig_file = os.path.join(
                        station_path,
                        f"{station_name}..{ch_name}__{event_name}_demeanDetrend_IC_BPF.mseed"
                    )

                    if not os.path.exists(orig_file):
                        print(f"⚠️ Missing original file: {orig_file}")
                        continue

                    # ----------------------------------------------------------------
                    # Read & cut waveform
                    # ----------------------------------------------------------------
                    try:
                        st = read(orig_file)
                        tr = st[0]
                        data = tr.data
                        sr = tr.stats.sampling_rate

                        segment = data[start_idx:end_peak_idx]
                        if len(segment) == 0:
                            print(f"⚠️ Empty segment in {orig_file}")
                            continue

                        # Create new stream
                        new_tr = tr.copy()
                        new_tr.data = segment
                        new_st = Stream([new_tr])

                    except Exception as e:
                        print(f"⚠️ Error reading {orig_file}: {e}")
                        continue

                    # ----------------------------------------------------------------
                    # Construct output filename
                    # ----------------------------------------------------------------
                    out_name = (
                        f"{station_name}..{ch_name}__{event_name}"
                        f"_demeanDetrend_IC_BPF__PS__duration_LE_{max_ps_duration}"
                        f"__SNR_GE_{min_snr}.mseed"
                    )
                    out_path = os.path.join(output_dir, out_name)

                    # Save output
                    try:
                        new_st.write(out_path, format="MSEED")
                        print(f"✅ Created: {out_path}")
                    except Exception as e:
                        print(f"❌ Error writing {out_path}: {e}")

    print(f"\n🎉 Completed PS file creation for SNR ≥ {min_snr}, PS_duration ≤ {max_ps_duration}s.\n")


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






def count_nof_training_stations_and_create_json_files(
        min_snr: float,
        max_ps_duration: float,
        depthMin: float,
        depthMax: float):

    """
    Classifies Z-channel signals from boundaries.json into:

    A) Training-eligible:
         - minimum_station_snr >= min_snr
         - peak_segment_duration_time <= max_ps_duration
         - clean_event_duration_time >= min_clean_event_duration
         - depthMin <= Depth <= depthMax

    B1) High SNR (>= min_snr) & TOO LONG peak segment (> max_ps_duration s) and depth in bounds
    B2) Low SNR (< min_snr)
    B3) High SNR (>= min_snr) & TOO SHORT clean event (< min_clean_event_duration s)
    B4) High SNR (>= min_snr) & (Depth < depthMin km or Depth > depthMax km)

    Produces:
      • trainingSet_SNR_GE_<min_snr>_PS_duration_LE_<max_ps_duration>.json
      • PotentiallyUsedOnTrainingSet_SNR_GE_<min_snr>_PS_duration_GE_<max_ps_duration>.json
    """

    import os
    from main import LOG_DIR, BASE_DIR

    # Threshold για clean_event_duration (ίσο με max_ps_duration όπως πριν)
    min_clean_event_duration = max_ps_duration

    # --- Paths ---
    json_path = os.path.join(LOG_DIR, "boundaries.json")

    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return

    data = load_json(json_path)

    # --- New JSON files ---
    training_json = {}
    potential_json = {}

    # --- Counters ---
    to_training = 0
    high_snr_and_high_ps_duration_and_depth_in_bounds = 0  # B1
    low_snr = 0                        # B2
    high_snr_but_low_clean_event = 0   # B3
    depth_out_of_range = 0             # B4

    # --- Traverse structure: year → event → station → channel ---
    for year, events in data.items():

        if year == "total_nof_stations":
            continue
        if not isinstance(events, dict):
            continue

        for event_name, stations in events.items():
            if not isinstance(stations, dict):
                continue

            # --------------------------------------------------
            # ΒΡΕΣ ΤΟ ΒΑΘΟΣ ΤΟΥ EVENT ΑΠΟ info.json
            # --------------------------------------------------
            depth_km = None
            info_path = os.path.join(BASE_DIR, str(year), event_name, "info.json")
            if os.path.exists(info_path):
                try:
                    info_data = load_json(info_path)
                    # Προσπάθησε με διαφορετικά πιθανά κλειδιά
                    depth_km = (
                        info_data.get("Depth_km")
                        or info_data.get("Depth-km")
                        or info_data.get("depth_km")
                        or info_data.get("depth-km")
                    )
                    if depth_km is not None:
                        depth_km = float(depth_km)
                except Exception as e:
                    print(f"⚠️ Could not read depth from {info_path}: {e}")
                    depth_km = None

            for station_name, channels in stations.items():
                if not isinstance(channels, dict):
                    continue

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
                    # CATEGORY B1 — HIGH SNR but PS too long
                    #   (Αυτά μπαίνουν και στο PotentiallyUsedOnTrainingSet.json)
                    #   Προσοχή: εδώ ΔΕΝ ελέγχουμε depth.
                    # ------------------------------------------------------
                    if ps_dur > max_ps_duration:
                        # Υποψήφιος ΜΟΝΟ αν depth είναι γνωστό και εντός ορίων
                        if depth_km is not None and depthMin <= depth_km <= depthMax:
                            high_snr_and_high_ps_duration_and_depth_in_bounds += 1

                            year_dict = potential_json.setdefault(year, {})
                            event_dict = year_dict.setdefault(event_name, {})
                            event_dict[station_name] = channels

                        # Αν depth είναι None ή εκτός ορίων → τότε πάει στο B4
                        else:
                            depth_out_of_range += 1

                        continue

                    # ------------------------------------------------------
                    # CATEGORY B4 — HIGH SNR & DEPTH OUT OF RANGE
                    #   Depth < depthMin ή Depth > depthMax
                    #   Αν depth_km is None → θεωρείται εκτός ορίων
                    # ------------------------------------------------------
                    if depth_km is None or depth_km < depthMin or depth_km > depthMax:
                        depth_out_of_range += 1
                        continue

                    # ------------------------------------------------------
                    # CATEGORY B3 — HIGH SNR but Clean Event too short
                    # ------------------------------------------------------
                    if clean_dur < min_clean_event_duration:
                        high_snr_but_low_clean_event += 1
                        continue

                    # ------------------------------------------------------
                    # CATEGORY A — Training-eligible
                    # ------------------------------------------------------
                    if depthMin <= depth_km <= depthMax:
                        to_training += 1

                    year_dict = training_json.setdefault(year, {})
                    event_dict = year_dict.setdefault(event_name, {})
                    event_dict[station_name] = channels

    # ----------------------------------------------------------
    # SAVE: TRAINING SET JSON
    # ----------------------------------------------------------
    output_name = f"trainingSet_SNR_GE_{min_snr}_PS_duration_LE_{max_ps_duration}_and_{depthMin}_LE_Depth_LE_{depthMax}.json"
    output_path = os.path.join(LOG_DIR, output_name)

    save_json(output_path, training_json)
    print(f"\n💾 Training Set JSON saved to:\n   {output_path}")
    print(f"📦 Contains {to_training} training-eligible stations.\n")

    # ----------------------------------------------------------
    # SAVE: POTENTIAL TRAINING SET JSON (B1)
    # ----------------------------------------------------------
    potential_path = os.path.join(
        LOG_DIR,
        f"PotentiallyUsedOnTrainingSet_SNR_GE_{min_snr}_PS_duration_GE_{max_ps_duration}_and_{depthMin}_LE_Depth_LE_{depthMax}.json"
    )
    save_json(potential_path, potential_json)

    print(f"💾 Potential Training Set JSON saved to:\n   {potential_path}")
    print(f"📦 Contains {high_snr_and_high_ps_duration_and_depth_in_bounds} potentially useful stations.\n")

    # ----------------------------------------------------------
    # Pretty print report
    # ----------------------------------------------------------
    def print_report_line(label, value, width=110):
        dots = "." * max(1, width - len(label))
        print(f"{label} {dots} {value:>6}")

    print("\n📊 *** SIGNAL CLASSIFICATION REPORT ***")

    # B1
    label1 = f"⚠ POTENCIALY USED : SNR ≥ {min_snr} & PS_duration_time > {max_ps_duration} sec & {depthMin}≤Depth≤{depthMax}"
    print_report_line(label1, high_snr_and_high_ps_duration_and_depth_in_bounds)

    # B2
    label2 = f"⚠ NOT USED SET : SNR < {min_snr}"
    print_report_line(label2, low_snr)

    # B3
    label3 = f"⚠ NOT USED SET : SNR ≥ {min_snr} & clean_event_duration < {min_clean_event_duration} sec"
    print_report_line(label3, high_snr_but_low_clean_event)

    # B4
    label4 = f"⚠ NOT USED SET : SNR ≥ {min_snr} & (Depth < {depthMin} km or Depth > {depthMax} km)"
    print_report_line(label4, depth_out_of_range)

    print("-" * 110)

    # TRAINING
    label5 = (
        f"✔ TRAINING SET : SNR ≥ {min_snr} & PS_duration_time ≤ {max_ps_duration} sec "
        f"& clean_event_duration ≥ {min_clean_event_duration} sec "
        f"& {depthMin} km ≤ Depth ≤ {depthMax} km"
    )
    print_report_line(label5, to_training)

def find_stations_for_ps_fixed(
        min_snr: float,
        max_ps_duration: float,
        min_event_duration: float,
        depth_min: float,
        depth_max: float):
    """
    Creates a PS-FIXED JSON structure by scanning boundaries.json
    and keeping ONLY the stations that satisfy ALL criteria:

    • minimum_station_snr >= min_snr
    • peak_segment_duration_time <= max_ps_duration
    • clean_event_duration_time >= min_event_duration
    • depth_min <= Depth_km <= depth_max  (Depth from info.json)

    Output:
        Logs/PSfixed_SNR_GE_<min_snr>_PS_LE_<max_ps_duration>_
             CE_GE_<min_event_duration>_DEPTH_<depth_min>_<depth_max>.json
    """

    import os
    from collections import OrderedDict
    from main import LOG_DIR, BASE_DIR

    boundaries_path = os.path.join(LOG_DIR, "boundaries.json")

    if not os.path.exists(boundaries_path):
        print(f"❌ File not found: {boundaries_path}")
        return

    db = load_json(boundaries_path)
    psfixed_json = {}

    print("\n🔍 Running Find PS Fixed...")

    # --- Traverse year → event → station ---
    for year, events in db.items():

        if not isinstance(events, dict):
            continue
        if year == "total_nof_stations":
            continue

        for event_name, stations in events.items():

            # --------- Load depth from info.json ---------
            depth_km = None
            info_path = os.path.join(BASE_DIR, str(year), event_name, "info.json")

            if os.path.exists(info_path):
                try:
                    info = load_json(info_path)
                    depth_km = (
                        info.get("depth_km")
                        or info.get("Depth_km")
                        or info.get("depth-km")
                        or info.get("Depth-km")
                    )
                    if depth_km is not None:
                        depth_km = float(depth_km)
                except:
                    depth_km = None

            # If no depth → skip event
            if depth_km is None:
                continue

            # Depth filter
            if not (depth_min <= depth_km <= depth_max):
                continue

            # Now check stations
            for station_name, channels in stations.items():

                if not isinstance(channels, dict):
                    continue

                # ---- SNR check ----
                station_snr = channels.get("minimum_station_snr")
                if station_snr is None:
                    continue
                station_snr = float(station_snr)

                if station_snr < min_snr:
                    continue

                # ---- Z-channel checks ----
                station_is_valid = False

                for ch_name, ch_info in channels.items():

                    if not isinstance(ch_info, dict):
                        continue
                    if not ch_name.endswith("Z"):
                        continue

                    # peak segmentation duration
                    ps = ch_info.get("peak_segment_duration_time")
                    if ps is None:
                        continue
                    try:
                        ps = float(ps)
                    except:
                        continue
                    if ps > max_ps_duration:
                        continue

                    # clean event duration
                    ce = ch_info.get("clean_event_duration_time")
                    if ce is None:
                        continue
                    try:
                        ce = float(ce)
                    except:
                        continue
                    if ce < min_event_duration:
                        continue

                    # If we reach here, channel is valid
                    station_is_valid = True
                    break

                # Save station subtree if valid
                if station_is_valid:
                    year_dict = psfixed_json.setdefault(year, {})
                    event_dict = year_dict.setdefault(event_name, {})
                    event_dict[station_name] = channels

    # ------------------------------------------------------------------
    # COUNT total number of Z-channels (AFTER filtering)
    # ------------------------------------------------------------------
    total_nof_stations = 0
    for year, events in psfixed_json.items():
        if year == "total_nof_stations":
            continue
        for event_name, stations in events.items():
            for station_name, channels in stations.items():
                for ch_name, ch_info in channels.items():
                    if isinstance(ch_info, dict) and ch_name.endswith("Z"):
                        total_nof_stations += 1

    # ----------------------------------------------------------
    # REORDER JSON SO total_nof_stations APPEARS FIRST
    # ----------------------------------------------------------
    ordered_output = OrderedDict()
    ordered_output["total_nof_stations"] = total_nof_stations

    for key, val in psfixed_json.items():
        if key != "total_nof_stations":
            ordered_output[key] = val

    # ----------------------------------------------------------
    # SAVE RESULT JSON
    # ----------------------------------------------------------
    output_name = (
        f"StationsForPsFixed_{min_snr}_{max_ps_duration}_"
        f"({min_event_duration}_({depth_min}-{depth_max}).json"
    )
    output_path = os.path.join(LOG_DIR, output_name)

    save_json(output_path, ordered_output)

    print(f"\n💾 PS-FIXED JSON saved to:")
    print(f"   {output_path}")

    # ----------------------------------------------------------
    # Count total stations
    # ----------------------------------------------------------
    total_stations = sum(
        len(stations) for (year, events) in psfixed_json.items()
        if isinstance(events, dict)
        for (event_name, stations) in events.items()
    )

    print(f"📦 Total stations included: {total_stations}")
    print(f"🎧 Total Z-channels included: {total_nof_stations}\n")

    return ordered_output

def find_stations_for_ps_variants_and_clean_events(
        min_snr: float,
        max_ps_duration: float,
        depth_min: float,
        depth_max: float):
    """
    Creates a PS-VARIANT JSON structure by scanning boundaries.json
    and keeping ONLY the stations that satisfy ALL criteria:

    • minimum_station_snr >= min_snr
    • peak_segment_duration_time <= max_ps_duration
    • depth_min <= Depth_km <= depth_max  (Depth from info.json)
    • At least one Z-channel satisfies the above

    Output:
        Logs/PSvariant_SNR_GE_<min_snr>_PS_LE_<max_ps_duration>_
             DEPTH_<depth_min>_<depth_max>.json
    """

    import os
    from collections import OrderedDict
    from main import LOG_DIR, BASE_DIR

    boundaries_path = os.path.join(LOG_DIR, "boundaries.json")

    if not os.path.exists(boundaries_path):
        print(f"❌ File not found: {boundaries_path}")
        return

    db = load_json(boundaries_path)
    psvariant_json = {}

    print("\n🔍 Running Find PS Variant...")

    # ------------------------------------------------------------
    # Traverse year → event → station
    # ------------------------------------------------------------
    for year, events in db.items():

        if not isinstance(events, dict):
            continue
        if year == "total_nof_stations":
            continue

        for event_name, stations in events.items():

            # --------- Load depth from info.json ---------
            depth_km = None
            info_path = os.path.join(BASE_DIR, str(year), event_name, "info.json")

            if os.path.exists(info_path):
                try:
                    info = load_json(info_path)
                    depth_km = (
                        info.get("depth_km")
                        or info.get("Depth_km")
                        or info.get("depth-km")
                        or info.get("Depth-km")
                    )
                    if depth_km is not None:
                        depth_km = float(depth_km)
                except:
                    depth_km = None

            if depth_km is None:
                continue

            # depth filtering
            if not (depth_min <= depth_km <= depth_max):
                continue

            # ------------------------------------------------------------
            # Now evaluate each station
            # ------------------------------------------------------------
            for station_name, channels in stations.items():

                if not isinstance(channels, dict):
                    continue

                # SNR check
                station_snr = channels.get("minimum_station_snr")
                if station_snr is None:
                    continue
                station_snr = float(station_snr)

                if station_snr < min_snr:
                    continue

                # Z-channel validation
                station_is_valid = False

                for ch_name, ch_info in channels.items():

                    if not isinstance(ch_info, dict):
                        continue
                    if not ch_name.endswith("Z"):
                        continue

                    # peak segmentation duration
                    ps = ch_info.get("peak_segment_duration_time")
                    if ps is None:
                        continue
                    try:
                        ps = float(ps)
                    except:
                        continue

                    if ps > max_ps_duration:
                        continue

                    # If reached here, channel is valid
                    station_is_valid = True
                    break

                if station_is_valid:
                    year_dict = psvariant_json.setdefault(year, {})
                    event_dict = year_dict.setdefault(event_name, {})
                    event_dict[station_name] = channels

    # ------------------------------------------------------------
    # Count total Z-channels in output dataset
    # ------------------------------------------------------------
    total_nof_stations = 0
    for year, events in psvariant_json.items():
        for event_name, stations in events.items():
            for station_name, channels in stations.items():
                for ch_name, ch_info in channels.items():
                    if isinstance(ch_info, dict) and ch_name.endswith("Z"):
                        total_nof_stations += 1

    # ------------------------------------------------------------
    # REORDER JSON so that total_nof_stations is FIRST
    # ------------------------------------------------------------
    ordered_output = OrderedDict()
    ordered_output["total_nof_stations"] = total_nof_stations

    for key, val in psvariant_json.items():
        ordered_output[key] = val

    # ------------------------------------------------------------
    # SAVE RESULT FILE
    # ------------------------------------------------------------
    output_name = (
        f"StationsForPsVariantsAndCleanEvents_{min_snr}_{max_ps_duration}_"
        f"({depth_min}-{depth_max}).json"
    )

    output_path = os.path.join(LOG_DIR, output_name)
    save_json(output_path, ordered_output)

    print(f"\n💾 PS-VARIANT JSON saved to:")
    print(f"   {output_path}")
    print(f"📦 Total Z-channels included: {total_nof_stations}\n")

    return ordered_output


# ==========================================================
if __name__ == "__main__":
    #find_boundaries()

    find_stations_for_ps_fixed(5, 30, 30, 1,24)
    find_stations_for_ps_variants_and_clean_events(5, 30, 1, 24)
    #create_cutted_signal_files(5, 30)
