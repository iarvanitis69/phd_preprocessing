#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Βήμα 1: Demean + Detrend + Αποθήκευση σε .vtu (VTK)
---------------------------------------------------
Διαβάζει όλα τα *.mseed αρχεία, εφαρμόζει προεπεξεργασία
και δημιουργεί νέο αρχείο .vtu ανά σταθμό/σεισμό, το οποίο
περιέχει και τα 3 κανάλια (π.χ. HHE, HHN, HHZ) ενωμένα.

Η έξοδος έχει μορφή:
  HL.SANT__20250125T065655Z__20250125T070025Z_demean_detrend.vtu

Αποθηκεύονται όλα τα traces μαζί (ένα για κάθε κανάλι).
"""

import os
import json
import numpy as np
from obspy import read


def write_error(error_path, event_dir, station, channel, message):
    """Καταγράφει σφάλματα σε JSON ανά event και σταθμό."""
    if os.path.exists(error_path):
        with open(error_path, "r", encoding="utf-8") as f:
            errors = json.load(f)
    else:
        errors = {}

    event_key = os.path.basename(os.path.dirname(os.path.dirname(event_dir)))
    errors.setdefault(event_key, {})
    errors[event_key].setdefault(station, {})
    errors[event_key][station][channel] = message

    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f"🛑 Σφάλμα: {event_key}/{station}/{channel} → {message}")


def demean_detrend():
    from main import BASE_DIR

    logs_dir = os.path.join(BASE_DIR, "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    error_path = os.path.join(logs_dir, "demean_detrend_errors.json")

    for root, _, files in os.walk(BASE_DIR):
        if "Logs" in root or "Stations" in root:
            continue

        # Βρες όλα τα .mseed του φακέλου (αντιστοιχούν σε 1 σταθμό/σεισμό)
        file_list = sorted(f for f in files if f.endswith(".mseed") and "_demean_detrend" not in f)
        if not file_list:
            continue

        try:
            # Δημιουργία τελικού ονόματος εξόδου χωρίς κανάλι
            first_file = file_list[0]
            parts = first_file.split("__")
            if len(parts) >= 3:
                station_full = ".".join(parts[0].split(".")[:2])  # π.χ. HL.SANT
                start_time = parts[1]
                end_time = parts[2].replace(".mseed", "")
                output_file = f"{station_full}__{start_time}__{end_time}_demean_detrend.vtu"
            else:
                raise Exception(f"Μη αναγνωρίσιμο αρχείο: {first_file}")

            output_path = os.path.join(root, output_file)
            if os.path.exists(output_path):
                print(f"⏩ Παράκαμψη (υπάρχει ήδη): {output_path}")
                continue

            all_points = []
            all_amplitudes = []
            all_labels = []

            for file in file_list:
                input_path = os.path.join(root, file)
                st = read(input_path)
                st.detrend("demean")
                st.detrend("linear")

                for tr in st:
                    data = tr.data.astype(np.float32)
                    times = np.linspace(0, len(data) / tr.stats.sampling_rate, num=len(data))

                    # Δημιουργία 1D γεωμετρίας (X = χρόνος)
                    points = np.zeros((len(data), 3))
                    points[:, 0] = times

                    all_points.append(points)
                    all_amplitudes.append(data)
                    all_labels.append(np.full(len(data), tr.stats.channel))  # π.χ. HHZ

            # Συνένωση όλων των καναλιών
            points = np.concatenate(all_points, axis=0)
            amplitude = np.concatenate(all_amplitudes, axis=0)
            channel_label = np.concatenate(all_labels, axis=0)

            pdata = pv.PolyData(points)
            pdata["amplitude"] = amplitude
            pdata["channel"] = channel_label

            pdata.save(output_path)
            print(f"✅ Αποθηκεύτηκε: {output_path}")

        except Exception as e:
            msg = f"Σφάλμα: {e}"
            write_error(error_path, root, "GENERAL", "GENERAL", msg)

