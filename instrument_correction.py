#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Βήμα 2: Instrument Correction σε *_demean_detrend.mseed
-------------------------------------------------------
Επεξεργάζεται δομή φακέλων:
  /media/iarv/Samsung/Events/<Year>/<Event>/<Station>/

✅ Αν υπάρχουν ήδη τα *_demean_detrend_IC.mseed αρχεία, παραλείπονται.
✅ Αντιμετωπίζει λάθος dtype / encoding, διορθώνει XMLs με NotEnumeratedValue.
✅ Γράφει FLOAT32 (encoding=3).
✅ Παραλείπει excluded σταθμούς (από excluded_stations.json).
"""

import os
import json
import numpy as np
from obspy import read, read_inventory


def load_excluded_stations():
    from main import LOG_DIR
    excluded_path = os.path.join(LOG_DIR, "excluded_stations.json")
    if os.path.exists(excluded_path):
        try:
            with open(excluded_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}


def is_station_excluded(event_name, station, excluded):
    if event_name in excluded:
        for net_station in excluded[event_name]:
            if "." in net_station:
                _, excluded_station = net_station.split(".")
                if excluded_station == station:
                    print(f"🚫 Παράκαμψη excluded σταθμού: {event_name}/{station}")
                    return True
    return False


def write_error(year, event, station, channel, message):
    from main import LOG_DIR
    error_path = os.path.join(LOG_DIR, "instrumentCorrection_errors.json")
    if os.path.exists(error_path):
        try:
            with open(error_path, "r", encoding="utf-8") as f:
                errors = json.load(f)
        except json.JSONDecodeError:
            errors = {}
    else:
        errors = {}
    errors.setdefault(year, {}).setdefault(event, {}).setdefault(station, {}).setdefault(channel, []).append(str(message))
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)


def validate_and_fix_inventory(inv, xml_path=None):
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


def is_instrument_corrected(station_dir, original_file):
    output_file = original_file.replace("_demeanDetrend.mseed", "_demeanDetrend_IC.mseed")
    output_path = os.path.join(station_dir, output_file)
    return os.path.exists(output_path)


def process_station_dir(station_dir, year, event, excluded):
    station = os.path.basename(station_dir)

    if is_station_excluded(event, station, excluded):
        write_error(year, event, station, "-", "Παράλειψη excluded σταθμού")
        return

    mseed_list = sorted(f for f in os.listdir(station_dir) if f.endswith("_demeanDetrend.mseed"))
    if not mseed_list:
        return

    xml_guess = os.path.join(station_dir, f"{station}.xml")

    for file in mseed_list:
        input_path = os.path.join(station_dir, file)

        if is_instrument_corrected(station_dir, file):
            print(f"⏩ Παράκαμψη (υπάρχει): {station}/{file}")
            continue

        output_path = input_path.replace("_demeanDetrend.mseed", "_demeanDetrend_IC.mseed")

        try:
            st = ensure_int32_encoding(input_path)
        except Exception as e:
            write_error(year, event, station, "UNKNOWN", f"Ανάγνωση {file}: {e}")
            continue

        for tr in list(st):
            net, sta, cha = tr.stats.network or "UNK", tr.stats.station or "UNK", tr.stats.channel or "UNK"
            xml1 = os.path.join(station_dir, f"{net}.{sta}.xml")
            xml_path = xml1 if os.path.exists(xml1) else xml_guess
            if not os.path.exists(xml_path):
                write_error(year, event, station, cha, f"Δεν βρέθηκε StationXML: {net}.{sta}")
                st.remove(tr)
                continue

            try:
                inv = read_inventory(xml_path)
                inv = validate_and_fix_inventory(inv, xml_path)
                tr.remove_response(inventory=inv, output="VEL", zero_mean=True, taper=True, taper_fraction=0.05)
                tr.data = np.nan_to_num(tr.data, nan=0.0).astype(np.float32)
            except Exception as e:
                write_error(year, event, station, cha, f"remove_response error: {e}")
                st.remove(tr)

        if len(st) == 0:
            continue

        try:
            st.write(output_path, format="MSEED")
            print(f"✅ {station}/{os.path.basename(output_path)}")
        except Exception as e:
            write_error(year, event, station, "WRITE_FAIL", f"Αποτυχία αποθήκευσης: {output_path}: {e}")


def instrument_correction():
    from main import BASE_DIR
    excluded = load_excluded_stations()

    for year in sorted(os.listdir(BASE_DIR)):
        year_dir = os.path.join(BASE_DIR, year)
        if not os.path.isdir(year_dir):
            continue
        for event in sorted(os.listdir(year_dir)):
            event_dir = os.path.join(year_dir, event)
            if not os.path.isdir(event_dir):
                continue
            for station in sorted(os.listdir(event_dir)):
                station_dir = os.path.join(event_dir, station)
                if os.path.isdir(station_dir):
                    process_station_dir(station_dir, year, event, excluded)
