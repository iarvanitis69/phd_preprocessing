#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Βήμα 1: Demean + Detrend
-------------------------
Διαβάζει όλα τα *.mseed αρχεία και δημιουργεί νέα αρχεία
με κατάληξη *_dmin_dtrend.msid χωρίς να τροποποιεί τα αρχικά.
"""

import os
import json
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

        for file in files:
            if not file.endswith(".mseed"):
                continue
            if "_demean_detrend" in file:
                continue

            input_path = os.path.join(root, file)
            base_name = os.path.splitext(file)[0]
            output_path = os.path.join(root, f"{base_name}_demean_detrend.mseed")

            if os.path.exists(output_path):
                print(f"⏩ Παράκαμψη (υπάρχει ήδη): {output_path}")
                continue

            try:
                st = read(input_path)
                print(f"📄 Επεξεργασία: {file}")
                st.detrend("demean")
                st.detrend("linear")
                for tr in st:
                    tr.data = tr.data.astype("int32")

                st.write(output_path, format="MSEED", encoding="STEIM2")
                print(f"✅ Αποθηκεύτηκε: {output_path}")
            except Exception as e:
                msg = f"Σφάλμα: {e}"
                write_error(error_path, root, "GENERAL", "GENERAL", msg)
