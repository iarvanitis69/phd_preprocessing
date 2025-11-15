#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from obspy import read

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

    # ---------------------------------------------------------
    # Helper: atomic save JSON
    # ---------------------------------------------------------
    def save_json(path, data):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------------------------------------------------------
    # Bandpass 1–20 Hz
    # ---------------------------------------------------------
    def bandpass_filter(data, sr, fmin=1.0, fmax=20.0, order=4):
        nyquist = 0.5 * sr
        b, a = butter(order, [fmin/nyquist, fmax/nyquist], btype="band")
        return filtfilt(b, a, data)

    # ---------------------------------------------------------
    # Load SNR JSON
    # ---------------------------------------------------------
    snr_json = os.path.join(LOG_DIR, "snr.json")
    if not os.path.exists(snr_json):
        print(f"❌ Δεν υπάρχει το {snr_json}")
        return

    try:
        snr_data = load_json(snr_json)
    except Exception as e:
        print(f"❌ Σφάλμα ανάγνωσης snr.json: {e}")
        return

    # ---------------------------------------------------------
    # Prepare boundaries_HHZ.json
    # NEW RULE: total_nof_stations ALWAYS on top level
    # ---------------------------------------------------------
    boundaries_path = os.path.join(LOG_DIR, "boundaries_HHZ.json")

    if os.path.exists(boundaries_path):
        try:
            all_results = load_json(boundaries_path)
            print("📂 Φορτώθηκε υπάρχον boundaries_HHZ.json")

            # Ensure top-level counter exists
            if "total_nof_stations" not in all_results:
                all_results = {"total_nof_stations": 0, **all_results}

            existing_counter = all_results.get("total_nof_stations", 0)

        except Exception:
            # corrupted json → reset
            all_results = {"total_nof_stations": 0}
            existing_counter = 0

    else:
        all_results = {"total_nof_stations": 0}
        existing_counter = 0

    AIC_FAIL_JSON = os.path.join(LOG_DIR, "AIC_failure.json")
    events_dict = snr_data.get("Events", {})

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------
    for year, events in events_dict.items():

        for eventJson, stations in events.items():

            for stationJson, chans in stations.items():

                # ---------------------------------------------------------
                # SKIP if station exists (continue from where we stopped)
                # ---------------------------------------------------------
                if (
                    str(year) in all_results
                    and eventJson in all_results[str(year)]
                    and stationJson in all_results[str(year)][eventJson]
                ):
                    print(f"⏭️ Παράλειψη (ήδη υπάρχει): {year}/{eventJson}/{stationJson}")
                    continue

                station_dict = {}
                station_snr = chans.get("minimum_snr", 0)

                # Paths
                event_path = os.path.join(BASE_DIR, str(year), eventJson)
                station_path = os.path.join(event_path, stationJson)

                if not os.path.exists(station_path):
                    continue

                # ---------------------------------------------------------
                # Scan files for HHN / HHE / HHZ
                # ---------------------------------------------------------
                for fname in os.listdir(station_path):
                    if not fname.endswith("_demeanDetrend_IC_BPF.mseed"):
                        continue
                    if not ("HHN" in fname or "HHE" in fname or "HHZ" in fname):
                        continue

                    file_path = os.path.join(station_path, fname)

                    try:
                        st = read(file_path)
                    except Exception as e:
                        print(f"⚠️ Αποτυχία ανάγνωσης {file_path}: {e}")
                        continue

                    for tr in st:
                        ch_id = tr.id.split(".")[-1]
                        sr = tr.stats.sampling_rate
                        data = tr.data.astype(float)

                        total_samples = len(data)
                        total_signal_time = total_samples / sr

                        # HHZ ONLY → store boundaries
                        if ch_id != "HHZ":
                            continue

                        # -------------------------------------------------
                        # AIC Picker
                        # -------------------------------------------------
                        start_of_event_HHZ_idx, _ = aic_picker(data)
                        if start_of_event_HHZ_idx is None:

                            try:
                                aic_failures = load_json(AIC_FAIL_JSON) if os.path.exists(AIC_FAIL_JSON) else {}
                            except Exception:
                                aic_failures = {}

                            year_dict = aic_failures.setdefault(str(year), {})
                            event_dict = year_dict.setdefault(eventJson, {})
                            if stationJson not in event_dict:
                                event_dict[stationJson] = True
                                aic_failures["count"] = aic_failures.get("count", 0) + 1

                            save_json(AIC_FAIL_JSON, aic_failures)
                            continue

                        # -------------------------------------------------
                        # Bandpass + Hilbert
                        # -------------------------------------------------
                        filtered = bandpass_filter(data, sr)
                        envelope = np.abs(hilbert(filtered))
                        norm_env = envelope / (np.max(envelope) or 1.0)

                        # -------------------------------------------------
                        # Peak detection
                        # -------------------------------------------------
                        buffer_samples = int(0.5 * sr)
                        search_segment = norm_env[start_of_event_HHZ_idx + buffer_samples:]
                        threshold = 0.2 * np.max(search_segment)

                        peaks, _ = find_peaks(
                            search_segment,
                            height=threshold,
                            prominence=0.8,
                            distance=int(0.3 * sr)
                        )

                        if len(peaks) == 0:
                            peak_amplitude_HHZ_idx = start_of_event_HHZ_idx + np.argmax(search_segment)
                        else:
                            peak_amplitude_HHZ_idx = start_of_event_HHZ_idx + peaks[0]

                        start_of_event_HHZ_datetime = tr.stats.starttime + start_of_event_HHZ_idx / sr
                        peak_amplitude_HHZ_datetime = tr.stats.starttime + peak_amplitude_HHZ_idx / sr

                        # -------------------------------------------------
                        # End of peak (symmetric)
                        # -------------------------------------------------
                        end_of_peak_segment_HHZ_idx = 2 * peak_amplitude_HHZ_idx - start_of_event_HHZ_idx
                        end_of_peak_segment_HHZ_datetime = tr.stats.starttime + end_of_peak_segment_HHZ_idx / sr

                        # -------------------------------------------------
                        # End of event based on SNR
                        # -------------------------------------------------
                        channel_snr = chans.get("HHZ", {}).get("snr", 1)
                        threshold_end = 1.0 / channel_snr
                        end_of_event_HHZ_idx = None

                        for i in range(peak_amplitude_HHZ_idx, len(norm_env)):
                            if norm_env[i] <= threshold_end:
                                end_of_event_HHZ_idx = i
                                break
                        if end_of_event_HHZ_idx is None:
                            end_of_event_HHZ_idx = len(norm_env) - 1

                        end_of_event_HHZ_idx += start_of_event_HHZ_idx
                        end_of_event_HHZ_time = tr.stats.starttime + end_of_event_HHZ_idx / sr

                        # Durations
                        clean_event_duration_HHZ_nof_samples = end_of_event_HHZ_idx - start_of_event_HHZ_idx
                        clean_event_duration_HHZ_time = clean_event_duration_HHZ_nof_samples / sr

                        peak_segment_duration_HHZ_nof_samples = end_of_peak_segment_HHZ_idx - start_of_event_HHZ_idx
                        peak_segment_duration_HHZ_time = peak_segment_duration_HHZ_nof_samples / sr

                        peak_amplitude_HHZ = float(norm_env[peak_amplitude_HHZ_idx])

                        # ---------------------------------------------------------
                        # SAVE HHZ-boundaries
                        # ---------------------------------------------------------
                        station_dict["start_of_event_HHZ_idx"] = int(start_of_event_HHZ_idx)
                        station_dict["start_of_event_HHZ_datetime"] = str(start_of_event_HHZ_datetime)

                        station_dict["peak_amplitude_HHZ_idx"] = int(peak_amplitude_HHZ_idx)
                        station_dict["peak_amplitude_HHZ_datetime"] = str(peak_amplitude_HHZ_datetime)
                        station_dict["peak_amplitude_HHZ"] = round(peak_amplitude_HHZ, 5)

                        station_dict["end_of_peak_segment_HHZ_idx"] = int(end_of_peak_segment_HHZ_idx)
                        station_dict["end_of_peak_segment_HHZ_datetime"] = str(end_of_peak_segment_HHZ_datetime)

                        station_dict["peak_segment_duration_HHZ_nof_samples"] = int(peak_segment_duration_HHZ_nof_samples)
                        station_dict["peak_segment_duration_HHZ_time"] = f"{peak_segment_duration_HHZ_time:.2f}"

                        station_dict["end_of_event_HHZ_idx"] = int(end_of_event_HHZ_idx)
                        station_dict["end_of_event_HHZ_time"] = str(end_of_event_HHZ_time)

                        station_dict["clean_event_duration_HHZ_nof_samples"] = int(clean_event_duration_HHZ_nof_samples)
                        station_dict["clean_event_duration_HHZ_time"] = f"{clean_event_duration_HHZ_time:.2f}"

                # ---------------------------------------------------------
                # Save station results
                # ---------------------------------------------------------
                if len(station_dict) > 0:

                    station_dict["minimum_station_snr"] = round(station_snr, 3)

                    if str(year) not in all_results:
                        all_results[str(year)] = {}

                    if eventJson not in all_results[str(year)]:
                        all_results[str(year)][eventJson] = {}

                    all_results[str(year)][eventJson][stationJson] = station_dict

                    # ---------------------------------------------------------
                    # INCREASE COUNTER ONLY FOR NEW STATIONS
                    # ---------------------------------------------------------
                    all_results["total_nof_stations"] = all_results.get("total_nof_stations", 0) + 1

                    # Always save
                    save_json(boundaries_path, all_results)

                    print(
                        f"💾 Saved {year}/{eventJson}/{stationJson}: "
                        f"SNR={station_snr:.2f}, peak_segment_duration_HHZ_time={peak_segment_duration_HHZ_time:.2f}, "
                        f"clean_event_duration_HHZ_time={clean_event_duration_HHZ_time:.2f}, total_signal_time={total_signal_time:.2f}"
                    )

    print(f"\n✅ Ολοκληρώθηκε. boundaries_HHZ.json ενημερώθηκε στο: {boundaries_path}")


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

# ==========================================================
if __name__ == "__main__":
    find_boundaries()


