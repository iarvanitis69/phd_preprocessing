#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import welch


def process_file(filepath, base_dir, log_dir):
    try:
        stream = read(filepath)
        trace = stream[0]
        data = trace.data

        # --- Welch PSD ---
        freqs_welch, power_welch = welch(data, fs=trace.stats.sampling_rate, nperseg=1024)
        max_energy_freq = freqs_welch[np.argmax(power_welch)]

        # --- Cutoff συχνότητα στο PSD: 95% ενέργεια ---
        cumulative_energy = np.cumsum(power_welch)
        total_energy = cumulative_energy[-1]
        target_energy = 0.95 * total_energy
        cutoff_index = np.searchsorted(cumulative_energy, target_energy)
        cutoff_freq = freqs_welch[min(cutoff_index, len(freqs_welch) - 1)]

        # --- FFT μόνο για το γράφημα ---
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), d=trace.stats.delta)
        power = np.abs(fft) ** 2

        # --- Γραφική Παράσταση ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(freqs, power, color='blue', label='FFT Power Spectrum')
        ax1.axvline(max_energy_freq, color='r', linestyle='--', label=f'Max PSD Energy: {max_energy_freq:.2f} Hz')
        ax1.axvline(cutoff_freq, color='g', linestyle='--', label=f'PSD Cutoff (95%): {cutoff_freq:.2f} Hz')
        ax1.set_ylabel('Power (FFT)')
        ax1.set_title(f"FFT Power Spectrum: {trace.stats.station}.{trace.stats.channel}")
        ax1.legend()

        ax2.plot(freqs_welch, power_welch, color='orange', label='PSD (Welch)', linewidth=2)
        ax2.axvline(max_energy_freq, color='r', linestyle='--')
        ax2.axvline(cutoff_freq, color='g', linestyle='--')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power (Welch PSD)')
        ax2.set_title("Power Spectral Density (Welch)")
        ax2.legend()

        plt.tight_layout()
        png_path = os.path.splitext(filepath)[0] + ".png"
        plt.savefig(png_path)
        plt.close()
        print(f"✅ Αποθηκεύτηκε στο: {png_path}")

        # --- Εγγραφή σε JSON ---
        json_path = os.path.join(log_dir, "fourier.json")
        rel_path = os.path.relpath(filepath, base_dir)
        parts = rel_path.split(os.sep)

        if len(parts) >= 4:
            event = parts[-4]
            station = parts[-3]
            channel = trace.stats.channel

            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    try:
                        data_json = json.load(f)
                    except json.JSONDecodeError:
                        data_json = {}
            else:
                data_json = {}

            data_json.setdefault(event, {}).setdefault(station, {})[channel] = {
                "max_energy_freq": round(float(max_energy_freq), 3),
                "cutoff_freq_95_percent": round(float(cutoff_freq), 3)
            }

            with open(json_path, "w") as f:
                json.dump(data_json, f, indent=2)

    except Exception as e:
        print(f"❌ Σφάλμα στο αρχείο {filepath}: {e}")


def find_max_freq():
    from main import BASE_DIR, LOG_DIR
    json_path = os.path.join(LOG_DIR, "fourier.json")
    os.makedirs(LOG_DIR, exist_ok=True)

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
    find_max_freq()
