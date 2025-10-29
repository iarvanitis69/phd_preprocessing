#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from obspy import read


def process_file(filepath):
    from main import BASE_DIR, LOG_DIR
    JSON_PATH = os.path.join(LOG_DIR, "fourier.json")

    try:
        # === Διαβάζουμε το σήμα ===
        stream = read(filepath)
        trace = stream[0]
        data = trace.data

        # === FFT ===
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), d=trace.stats.delta)
        power = np.abs(fft) ** 2

        if np.max(power) == 0 or len(freqs) < 10:
            print(f"⚠️ Χωρίς ενέργεια: {filepath}")
            return

        # === Ενέργεια ===
        max_power = np.max(power)
        max_energy_idx = np.argmax(power)
        max_energy_freq = freqs[max_energy_idx]
        threshold = 0.05 * max_power

        # === Cutoff από 50Hz προς τα αριστερά ===
        cutoff_freq = freqs[-1]
        for i in range(len(freqs) - 1, -1, -1):
            if power[i] >= threshold:
                cutoff_freq = freqs[i]
                break

        # === Γράφημα ===
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, power, label="Power Spectrum")
        plt.axvline(max_energy_freq, color='r', linestyle='--', label=f"Max Energy: {max_energy_freq:.2f} Hz")
        plt.axvline(cutoff_freq, color='g', linestyle='--', label=f"Cutoff (5%): {cutoff_freq:.2f} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title(f"Spectrum: {trace.stats.station}.{trace.stats.channel}")
        plt.legend()
        png_path = os.path.splitext(filepath)[0] + ".png"
        plt.savefig(png_path)
        plt.close()

        # === Προετοιμασία κλειδιών για JSON ===
        rel_path = os.path.relpath(filepath, BASE_DIR)
        parts = rel_path.split(os.sep)
        if len(parts) < 4:
            print(f"⚠️ Δεν εντοπίστηκε σωστά το event_id για {filepath}")
            return

        event_id = parts[1]  # π.χ. 20100507T041515_36.68_25.71_15.0km_M3.4
        station = trace.stats.station
        channel = trace.stats.channel

        # === Φόρτωση υπάρχοντος JSON ===
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, "r") as f:
                try:
                    data_json = json.load(f)
                except json.JSONDecodeError:
                    data_json = {}
        else:
            data_json = {}

        # === Ενημέρωση εγγραφής ===
        if event_id not in data_json:
            data_json[event_id] = {}
        if station not in data_json[event_id]:
            data_json[event_id][station] = {}

        data_json[event_id][station][channel] = {
            "max_energy": format(max_power, ".3e"),
            "max_energy_freq": round(float(max_energy_freq), 3),
            "cutoff_freq_5_percent": round(float(cutoff_freq), 3)
        }

        # === Εύρεση όλων των cutoff για min/max ===
        cutoff_values = []
        for ev_key, ev_val in data_json.items():
            if not isinstance(ev_val, dict):
                continue
            for stat in ev_val:
                for chan in ev_val[stat]:
                    cutoff = ev_val[stat][chan].get("cutoff_freq_5_percent")
                    if isinstance(cutoff, (int, float)):
                        cutoff_values.append(cutoff)

        if cutoff_values:
            data_json["minCutoffFreq"] = round(min(cutoff_values), 3)
            data_json["maxCutoffFreq"] = round(max(cutoff_values), 3)

        # === Αποθήκευση ===
        with open(JSON_PATH, "w") as f:
            json.dump(data_json, f, indent=2)

        print(f"✅ {event_id} | {station}.{channel} | max: {format(max_power, '.3e')} | cutoff: {cutoff_freq:.2f} Hz")

    except Exception as e:
        print(f"❌ Σφάλμα στο αρχείο {filepath}: {e}")

def find_max_freq():
    from main import BASE_DIR
    for root, _, files in os.walk(BASE_DIR):
        for fname in files:
            if fname.endswith("_demean_detrend.mseed"):
                full_path = os.path.join(root, fname)
                process_file(full_path)
