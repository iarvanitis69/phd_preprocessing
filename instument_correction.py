#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Βήμα 2: Instrument Correction
-----------------------------
Διαβάζει τα *_dmin_dtrend.msid αρχεία και δημιουργεί νέα
με κατάληξη *_dmin_dtrend_instrument_correction.msid χωρίς να
τροποποιεί τα αρχικά.
"""

import os
import json
from obspy import read, read_inventory, UTCDateTime


def write_error(error_path, event_dir, station, channel, message, net_code=None):
    """Καταγράφει σφάλματα σε JSON με πλήρες όνομα σταθμού."""
    if os.path.exists(error_path):
        with open(error_path, "r", encoding="utf-8") as f:
            errors = json.load(f)
    else:
        errors = {}

    event_key = os.path.basename(os.path.dirname(os.path.dirname(event_dir)))
    net_sta_key = f"{net_code}.{station}" if net_code else station
    errors.setdefault(event_key, {})
    errors[event_key].setdefault(net_sta_key, {})
    errors[event_key][net_sta_key][channel] = message

    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f"🛑 {event_key}/{net_sta_key}/{channel} → {message}")


def instrument_correction_all():
    from main import BASE_DIR

    logs_dir = os.path.join(BASE_DIR, "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    error_path = os.path.join(logs_dir, "instrument_correction_errors.json")

    xml_dir = os.path.join(BASE_DIR, "Stations")
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]

    for root, _, files in os.walk(BASE_DIR):
        if "Logs" in root or "Stations" in root:
            continue

        for file in files:
            if not file.endswith("_demean_detrend.msid"):
                continue

            input_path = os.path.join(root, file)
            output_path = input_path.replace(
                "_demean_detrend.msid", "_demean_detrend_instCorrection.mseed"
            )

            if os.path.exists(output_path):
                print(f"⏩ Παράκαμψη (υπάρχει ήδη): {output_path}")
                continue

            try:
                st = read(input_path)
                corrected = []

                for tr in st:
                    net_code = tr.stats.network
                    sta_code = tr.stats.station
                    loc_code = tr.stats.location.strip()
                    cha_code = tr.stats.channel

                    target_xml = f"{net_code}.{sta_code}.xml"
                    xml_match = [f for f in xml_files if f.lower() == target_xml.lower()]
                    if not xml_match:
                        write_error(error_path, root, sta_code, cha_code,
                                    f"Δεν βρέθηκε XML για {net_code}.{sta_code}", net_code)
                        continue

                    inv_path = os.path.join(xml_dir, xml_match[0])
                    inventory = read_inventory(inv_path)

                    try:
                        inv_sel = inventory.select(network=net_code, station=sta_code,
                                                   location=loc_code, channel=cha_code,
                                                   time=tr.stats.starttime)
                        _ = inv_sel.get_response(tr.id, tr.stats.starttime)

                        tr.remove_response(inventory=inv_sel, output="VEL",
                                           zero_mean=False, taper=False)
                        tr.data = tr.data.astype("float32")
                        tr.stats.mseed = {"encoding": "FLOAT32"}
                        corrected.append(tr)
                    except Exception as e:
                        write_error(error_path, root, sta_code, cha_code,
                                    f"Σφάλμα correction: {e}", net_code)
                        continue

                if corrected:
                    from obspy import Stream
                    Stream(corrected).write(output_path, format="MSEED", encoding="STEIM2")
                    print(f"✅ Αποθηκεύτηκε: {output_path}")
                else:
                    print(f"⚠️ Δεν δημιουργήθηκε έξοδος για {file}")

            except Exception as e:
                write_error(error_path, root, "GENERAL", "GENERAL", f"Σφάλμα στο αρχείο: {e}", "GEN")


if __name__ == "__main__":
    instrument_correction_all()
