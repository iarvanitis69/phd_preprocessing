#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Βήμα 2: Instrument Correction
-----------------------------
Διαβάζει όλα τα *_demean_detrend.mseed αρχεία και δημιουργεί
αντίστοιχα *_demean_detrend_instCorrection.mseed χωρίς να
τροποποιεί τα αρχικά.

Καταγράφει αναλυτικά όλα τα σφάλματα στο Logs/InstrumentCorrectionError.json
με ιεραρχία:
  event → network.station → channel → μήνυμα σφάλματος

Προσαρμογή για νέα δομή:
- Τα StationXML αρχεία (.xml) βρίσκονται μέσα στον φάκελο κάθε σταθμού,
  όχι πλέον στον κοινό φάκελο Stations/.
"""

import os
import json
import numpy as np
from obspy import read, read_inventory


def write_error(error_path, event_dir, station, channel, message, net_code=None):
    """
    Καταγράφει σφάλματα σε JSON με καθαρό μήνυμα.
    Δομή: event → network.station → channel → μήνυμα
    """
    clean_message = " ".join(str(message).split())

    # Αν υπάρχει ήδη, φόρτωσε
    if os.path.exists(error_path):
        try:
            with open(error_path, "r", encoding="utf-8") as f:
                errors = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Προειδοποίηση: Το {os.path.basename(error_path)} ήταν κατεστραμμένο ή άδειο – δημιουργείται νέο.")
            errors = {}
    else:
        errors = {}

    # Προσδιορισμός event από τη νέα δομή
    # BASE/YEAR/EVENT/STATION → event είναι ο φάκελος 3 επίπεδα πάνω
    event_key = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(event_dir))))
    net_sta_key = f"{net_code}.{station}" if net_code else station

    errors.setdefault(event_key, {})
    errors[event_key].setdefault(net_sta_key, {})

    if channel in errors[event_key][net_sta_key]:
        if errors[event_key][net_sta_key][channel] == clean_message:
            return

    errors[event_key][net_sta_key][channel] = clean_message

    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False, sort_keys=True)


def instrument_correction():
    # -----------------------------
    # Logs setup
    # -----------------------------
    from main import BASE_DIR
    logs_dir = os.path.join(BASE_DIR, "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    error_path = os.path.join(logs_dir, "InstrumentCorrectionError.json")

    # -----------------------------
    # Επεξεργασία αρχείων
    # -----------------------------
    for root, _, files in os.walk(BASE_DIR):
        if "Logs" in root:
            continue

        for file in files:
            if not file.endswith("_demean_detrend.mseed"):
                continue

            input_path = os.path.join(root, file)
            output_path = input_path.replace(
                "_demean_detrend.mseed", "_demean_detrend_instCorrection.mseed")

            if os.path.exists(output_path):
                print(f"⏩ Παράκαμψη (υπάρχει ήδη): {output_path}")
                continue

            try:
                st = read(input_path)
                corrected = []

                # Ο φάκελος σταθμού περιέχει το XML του
                station_dir = os.path.dirname(input_path)
                xml_files = [f for f in os.listdir(station_dir) if f.endswith(".xml")]

                for tr in st:
                    net_code = tr.stats.network
                    sta_code = tr.stats.station
                    loc_code = tr.stats.location.strip()
                    cha_code = tr.stats.channel

                    # Αναζήτηση XML μέσα στον ίδιο φάκελο
                    target_xml = f"{net_code}.{sta_code}.xml"
                    xml_match = [f for f in xml_files if f.lower() == target_xml.lower()]
                    if not xml_match:
                        write_error(error_path, root, sta_code, cha_code,
                                    f"Δεν βρέθηκε StationXML για {net_code}.{sta_code}", net_code)
                        continue

                    xml_path = os.path.join(station_dir, xml_match[0])

                    try:
                        inventory = read_inventory(xml_path)
                        inv_sel = inventory.select(network=net_code, station=sta_code,
                                                   location=loc_code, channel=cha_code,
                                                   time=tr.stats.starttime)
                        _ = inv_sel.get_response(tr.id, tr.stats.starttime)

                        tr.remove_response(inventory=inv_sel, output="VEL",
                                           zero_mean=False, taper=False)

                        tr.data = tr.data.astype(np.float32)
                        corrected.append(tr)

                    except Exception as e:
                        write_error(error_path, root, sta_code, cha_code,
                                    f"Σφάλμα διόρθωσης οργάνου: {e}", net_code)
                        continue

                if corrected:
                    from obspy import Stream
                    Stream(corrected).write(output_path, format="MSEED", encoding="FLOAT32")
                    print(f"✅ Ολοκληρώθηκε: {output_path}")
                else:
                    msg = "⚠️ Καμία επιτυχής διόρθωση καναλιού – δεν δημιουργήθηκε έξοδος."
                    print(msg)

            except Exception as e:
                msg = f"❌ Αποτυχία επεξεργασίας αρχείου ({file}): {e}"
                print(msg)
                event_key = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(root))))
                write_error(error_path, root, "UnknownStation", "UnknownChannel", msg, "Unknown")


def main():
    instrument_correction()


if __name__ == "__main__":
    main()
