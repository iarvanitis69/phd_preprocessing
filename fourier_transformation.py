#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import welch

def extract_event_info(event_folder_name: str):
    try:
        parts = event_folder_name.split("_")
        origin_time = parts[0]
        lat = parts[1]
        lon = parts[2]
        depth = parts[3].replace("km", "")
        mag = parts[4].replace("M", "")
        return {
            "event_folder": event_folder_name,
            "origin_time": origin_time,
            "latitude": float(lat),
            "longitude": float(lon),
            "depth_km": float(depth),
            "magnitude": float(mag)
        }
    except Exception:
        return {
            "event_folder": event_folder_name,
            "origin_time": None,
            "latitude": None,
            "longitude": None,
            "depth_km": None,
            "magnitude": None
        }

def process_file(filepath, base_dir, log_dir):
    try:
        stream = read(filepath)
        trace = stream[0]
        data = trace.data
        fs = trace.stats.sampling_rate

        # --- Welch PSD ---
        freqs_welch, power_welch = welch(data, fs=fs, nperseg=1024)
        cumulative_energy = np.cumsum(power_welch)
        total_energy = cumulative_energy[-1]

        # --- Cutoff συχνότητες ---
        cutoff_index_95 = np.searchsorted(cumulative_energy, 0.95 * total_energy)
        cutoff_index_05 = np.searchsorted(cumulative_energy, 0.05 * total_energy)

        cutoff_freq_95 = freqs_welch[min(cutoff_index_95, len(freqs_welch) - 1)]
        cutoff_freq_05 = freqs_welch[min(cutoff_index_05, len(freqs_welch) - 1)]
        max_energy_freq = freqs_welch[np.argmax(power_welch)]

        # --- FFT για γράφημα ---
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), d=1/fs)
        power = np.abs(fft) ** 2

        # --- Γράφημα ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(freqs, power, color='blue', label='FFT Power Spectrum')
        ax1.axvline(max_energy_freq, color='r', linestyle='--', label=f'Max PSD Energy: {max_energy_freq:.2f} Hz')
        ax1.axvline(cutoff_freq_05, color='purple', linestyle='--', label=f'5% Cutoff: {cutoff_freq_05:.2f} Hz')
        ax1.axvline(cutoff_freq_95, color='g', linestyle='--', label=f'95% Cutoff: {cutoff_freq_95:.2f} Hz')
        ax1.set_ylabel('Power (FFT)')
        ax1.set_title(f"FFT: {trace.stats.station}.{trace.stats.channel}")
        ax1.legend()

        ax2.plot(freqs_welch, power_welch, color='orange', label='PSD (Welch)')
        ax2.axvline(max_energy_freq, color='r', linestyle='--')
        ax2.axvline(cutoff_freq_05, color='purple', linestyle='--')
        ax2.axvline(cutoff_freq_95, color='g', linestyle='--')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power (Welch PSD)')
        ax2.set_title("Welch PSD")
        ax2.legend()

        plt.tight_layout()
        png_path = os.path.splitext(filepath)[0] + ".png"
        plt.savefig(png_path)
        plt.close()
        print(f"✅ Αποθηκεύτηκε το γράφημα: {png_path}")

        # --- Ενημέρωση JSON ---
        json_path = os.path.join(log_dir, "fourier.json")
        rel_path = os.path.relpath(filepath, base_dir)
        parts = rel_path.split(os.sep)

        # Βρες φάκελο event με "_"
        event = next((p for p in parts if "_" in p and "T" in p), None)
        if not event:
            print(f"⚠️ Δεν βρέθηκε φάκελος event για το αρχείο: {filepath}")
            return

        station = trace.stats.station
        network = trace.stats.network
        net_sta = f"{network}.{station}"
        channel = trace.stats.channel

        # Ανάγνωση υπάρχοντος JSON
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                try:
                    data_json = json.load(f)
                except json.JSONDecodeError:
                    data_json = {}
        else:
            data_json = {}

        data_json.setdefault(event, {})

        # Καταχώριση καναλιού
        data_json[event].setdefault(net_sta, {})[channel] = {
            "max_energy_freq": round(float(max_energy_freq), 3),
            "cutoff_freq_95_percent": round(float(cutoff_freq_95), 3),
            "cutoff_freq_5_percent": round(float(cutoff_freq_05), 3)
        }

        # Ενημέρωση global min/max
        prev_max = data_json.get("maxUpperCutoffFreq", 0)
        prev_min = data_json.get("minLowerCutoffFreq", float("inf"))

        data_json["maxUpperCutoffFreq"] = round(max(prev_max, cutoff_freq_95), 3)
        data_json["minLowerCutoffFreq"] = round(min(prev_min, cutoff_freq_05), 3)

        with open(json_path, "w") as f:
            json.dump(data_json, f, indent=2)

    except Exception as e:
        print(f"❌ Σφάλμα στο αρχείο {filepath}: {e}")





def find_max_and_min_freq():
    from main import LOG_DIR
    json_path = os.path.join(LOG_DIR, "fourier.json")
    os.makedirs(LOG_DIR, exist_ok=True)

    from main import BASE_DIR
    for root, _, files in os.walk(BASE_DIR):
        for fname in files:
            if fname.endswith("_demean_detrend_IC.mseed"):
                full_path = os.path.join(root, fname)
                process_file(full_path, BASE_DIR, LOG_DIR)

    # Υπολογισμός min/max cutoff
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                data_json = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Το JSON ήταν άδειο ή κατεστραμμένο.")
                return

        cutoff_values = []
        for ev_val in data_json.values():
            if isinstance(ev_val, dict):
                for stat in ev_val:
                    for chan in ev_val[stat]:
                        cutoff = ev_val[stat][chan].get("cutoff_freq_95_percent")
                        if isinstance(cutoff, (int, float)):
                            cutoff_values.append(cutoff)

        if cutoff_values:
            data_json["minCutoffFreq"] = round(min(cutoff_values), 3)
            data_json["maxCutoffFreq"] = round(max(cutoff_values), 3)

            with open(json_path, "w") as f:
                json.dump(data_json, f, indent=2)

            print(f"\n📊 min: {min(cutoff_values):.2f} Hz | max: {max(cutoff_values):.2f} Hz")
            print(f"💾 Αποθηκεύτηκαν στο: {json_path}")

# ==========================================================
if __name__ == "__main__":
    find_max_and_min_freq()