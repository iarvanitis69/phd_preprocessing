#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Βήμα 2: Instrument Correction σε *_demean_detrend.mseed
-------------------------------------------------------
Επεξεργάζεται δομή φακέλων:
  /media/iarv/Samsung/Events/<Year>/<Event>/<Station>/

✅ Αν υπάρχουν ήδη όλα τα *_demean_detrend_IC.mseed αρχεία σε ένα σταθμό,
   ο σταθμός παραλείπεται (resume-friendly).
✅ Αν υπάρχουν μερικά, συνεχίζει μόνο για τα υπόλοιπα.
✅ Αντιμετωπίζει λάθος dtype / encoding, διορθώνει XMLs με NotEnumeratedValue.
✅ Γράφει FLOAT32 (encoding=3).
"""

import os
import json
import numpy as np
from obspy import read, read_inventory

# ---------------------------------------------------------
BASE_DIR = "/media/iarv/Samsung"
EVENTS_DIR = os.path.join(BASE_DIR, "Events")
LOGS_DIR = os.path.join(BASE_DIR, "Logs")
os.makedirs(LOGS_DIR, exist_ok=True)
ERROR_PATH = os.path.join(LOGS_DIR, "InstrumentCorrectionError.json")
# ---------------------------------------------------------


def write_error(year, event, station, channel, message):
    """Καταγράφει σφάλματα σε JSON με ιεραρχία Year→Event→Station→Channel."""
    if os.path.exists(ERROR_PATH):
        try:
            with open(ERROR_PATH, "r", encoding="utf-8") as f:
                errors = json.load(f)
        except json.JSONDecodeError:
            errors = {}
    else:
        errors = {}
    errors.setdefault(year, {}).setdefault(event, {}).setdefault(station, {}).setdefault(channel, []).append(str(message))
    with open(ERROR_PATH, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)


def validate_and_fix_inventory(inv, xml_path=None):
    """Διορθώνει NotEnumeratedValue σε COUNTS/M/S."""
    modified = False
    for net in inv:
        for sta in net:
            for cha in sta:
                if not hasattr(cha, "response") or cha.response is None:
                    continue
                for stage in cha.response.response_stages:
                    if getattr(stage, "input_units", None) == "NotEnumeratedValue":
                        stage.input_units = "COUNTS"
                        modified = True
                    if getattr(stage, "output_units", None) == "NotEnumeratedValue":
                        stage.output_units = "M/S"
                        modified = True
                sens = getattr(cha.response, "instrument_sensitivity", None)
                if sens:
                    if getattr(sens, "input_units", None) == "NotEnumeratedValue":
                        sens.input_units = "COUNTS"
                        modified = True
                    if getattr(sens, "output_units", None) == "NotEnumeratedValue":
                        sens.output_units = "M/S"
                        modified = True
    if modified and xml_path:
        print(f"⚙️ Διορθώθηκε StationXML: {xml_path}")
    return inv


def ensure_int32_encoding(input_path):
    """Αν το αρχείο έχει λάθος dtype για Steim2, επανακωδικοποιείται σε INT32 προσωρινά."""
    try:
        return read(input_path)
    except Exception as e:
        if "Wrong dtype" in str(e):
            print(f"⚠️ {os.path.basename(input_path)}: επανακωδικοποίηση σε Steim2 INT32...")
            st_raw = read(input_path, decode_data=False)
            for tr in st_raw:
                tr.data = np.asarray(tr.data, dtype=np.int32)
            fixed = input_path + ".tmp"
            st_raw.write(fixed, format="MSEED", encoding=11)
            return read(fixed)
        raise


def station_is_complete(station_dir):
    """Αν υπάρχουν ήδη και τα τρία *_demean_detrend_IC.mseed (HHE, HHN, HHZ), επιστρέφει True."""
    ic_files = [f for f in os.listdir(station_dir) if f.endswith("_demean_detrend_IC.mseed")]
    if len(ic_files) >= 3:  # θεωρούμε πλήρη όταν υπάρχουν 3 κανάλια
        return True
    return False


def process_station_dir(station_dir, year, event):
    """Επεξεργάζεται όλα τα *_demean_detrend.mseed σε έναν σταθμό."""
    station = os.path.basename(station_dir)

    # Αν υπάρχουν ήδη όλα τα IC → skip station
    if station_is_complete(station_dir):
        print(f"⏩ Ο σταθμός {station} είναι ήδη πλήρης, παράκαμψη.")
        return

    mseed_list = sorted(f for f in os.listdir(station_dir) if f.endswith("_demean_detrend.mseed"))
    if not mseed_list:
        return

    xml_guess = os.path.join(station_dir, f"{station}.xml")

    for file in mseed_list:
        input_path = os.path.join(station_dir, file)
        output_path = input_path.replace("_demean_detrend.mseed", "_demean_detrend_IC.mseed")

        if os.path.exists(output_path):
            print(f"⏩ Παράκαμψη (υπάρχει ήδη): {output_path}")
            continue

        try:
            st = ensure_int32_encoding(input_path)
        except Exception as e:
            write_error(year, event, station, "UNKNOWN", f"Αποτυχία ανάγνωσης {file}: {e}")
            continue

        for tr in list(st):
            net, sta, cha = tr.stats.network or "UNK", tr.stats.station or "UNK", tr.stats.channel or "UNK"
            xml1 = os.path.join(station_dir, f"{net}.{sta}.xml")
            xml_path = xml1 if os.path.exists(xml1) else xml_guess
            if not os.path.exists(xml_path):
                write_error(year, event, station, cha, f"Δεν βρέθηκε StationXML για {net}.{sta}")
                st.remove(tr)
                continue

            try:
                inv = read_inventory(xml_path)
                inv = validate_and_fix_inventory(inv, xml_path)
                tr.remove_response(inventory=inv, output="VEL", zero_mean=True, taper=True, taper_fraction=0.05)
                tr.data = np.nan_to_num(tr.data, nan=0.0).astype(np.float32)
            except Exception as e:
                write_error(year, event, station, cha, f"Αποτυχία remove_response: {e}")
                st.remove(tr)

        if len(st) == 0:
            continue

        try:
            st.write(output_path, format="MSEED")
            print(f"✅ {station}/{os.path.basename(output_path)}")
        except Exception as e:
            write_error(year, event, station, "WRITE_FAIL", f"Αποτυχία αποθήκευσης {output_path}: {e}")


def instrument_correction():
    """Διατρέχει όλη τη δομή Events/<Year>/<Event>/<Station>/"""
    for year in sorted(os.listdir(EVENTS_DIR)):
        year_dir = os.path.join(EVENTS_DIR, year)
        if not os.path.isdir(year_dir):
            continue

        for event in sorted(os.listdir(year_dir)):
            event_dir = os.path.join(year_dir, event)
            if not os.path.isdir(event_dir):
                continue

            for station in sorted(os.listdir(event_dir)):
                station_dir = os.path.join(event_dir, station)
                if not os.path.isdir(station_dir):
                    continue

                process_station_dir(station_dir, year, event)


if __name__ == "__main__":
    instrument_correction()
