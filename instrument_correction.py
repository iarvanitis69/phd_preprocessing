#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Βήμα 2: Instrument Correction σε .VTU αρχεία
--------------------------------------------
Διαβάζει κάθε *_demean_detrend.vtu (που περιέχει όλα τα κανάλια),
κάνει διόρθωση οργάνου χρησιμοποιώντας το αντίστοιχο StationXML
και αποθηκεύει το αποτέλεσμα ως:
  *_demean_detrend_instCorrection.vtu
"""

import os
import json
import numpy as np
import pyvista as pv
from obspy import Trace, Stream, UTCDateTime, read_inventory


def write_error(error_path, event_dir, station, channel, message, net_code=None):
    """Καταγραφή σφάλματος σε JSON."""
    clean_message = " ".join(str(message).split())

    if os.path.exists(error_path):
        try:
            with open(error_path, "r", encoding="utf-8") as f:
                errors = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Προειδοποίηση: Το {os.path.basename(error_path)} ήταν άδειο – δημιουργείται νέο.")
            errors = {}
    else:
        errors = {}

    event_key = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(event_dir))))
    net_sta_key = f"{net_code}.{station}" if net_code else station

    errors.setdefault(event_key, {})
    errors[event_key].setdefault(net_sta_key, {})
    errors[event_key][net_sta_key][channel] = clean_message

    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False, sort_keys=True)


def instrument_correction_vtu():
    """Διόρθωση οργάνου για όλα τα .vtu που περιέχουν τα 3 κανάλια μαζί."""
    from main import BASE_DIR

    logs_dir = os.path.join(BASE_DIR, "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    error_path = os.path.join(logs_dir, "InstrumentCorrectionError.json")

    for root, _, files in os.walk(BASE_DIR):
        if "Logs" in root:
            continue

        for file in files:
            if not file.endswith("_demean_detrend.vtu"):
                continue

            input_path = os.path.join(root, file)
            output_path = input_path.replace(
                "_demean_detrend.vtu", "_demean_detrend_instCorrection.vtu"
            )

            if os.path.exists(output_path):
                print(f"⏩ Παράκαμψη (υπάρχει ήδη): {output_path}")
                continue

            try:
                pdata = pv.read(input_path)
                channels = [key for key in pdata.point_data.keys() if key.startswith("HH")]
                if not channels:
                    raise ValueError("Δεν εντοπίστηκαν πεδία καναλιών (HHE/HHN/HHZ) στο VTU αρχείο.")

                # Κοινός χρόνος
                times = pdata.points[:, 0]
                dt = np.mean(np.diff(times))
                sampling_rate = 1.0 / dt
                start_time = UTCDateTime(0)  # Δεν έχουμε absolute χρόνο στο VTU

                # Εξαγωγή στοιχείων σταθμού από το filename
                # π.χ. HL.SANT__20250125T065655Z__20250125T070025Z_demean_detrend.vtu
                base = os.path.splitext(file)[0]
                parts = base.split("__")
                if len(parts) >= 3:
                    net_sta = parts[0]  # π.χ. HL.SANT
                    net_code, sta_code = net_sta.split(".")
                else:
                    net_code, sta_code = "XX", "STAT"

                loc_code = ""

                # Αναζήτηση StationXML στον ίδιο φάκελο
                station_dir = os.path.dirname(input_path)
                xml_files = [f for f in os.listdir(station_dir) if f.endswith(".xml")]
                target_xml = f"{net_code}.{sta_code}.xml"
                xml_match = [f for f in xml_files if f.lower() == target_xml.lower()]
                if not xml_match:
                    write_error(error_path, root, sta_code, "ALL",
                                f"Δεν βρέθηκε StationXML για {net_code}.{sta_code}", net_code)
                    continue

                xml_path = os.path.join(station_dir, xml_match[0])
                inventory = read_inventory(xml_path)

                corrected_channels = {}

                for chan in channels:
                    try:
                        data = pdata[chan]
                        tr = Trace(data=data.astype(np.float32))
                        tr.stats.network = net_code
                        tr.stats.station = sta_code
                        tr.stats.channel = chan
                        tr.stats.location = loc_code
                        tr.stats.starttime = start_time
                        tr.stats.sampling_rate = sampling_rate

                        inv_sel = inventory.select(network=net_code, station=sta_code,
                                                   location=loc_code, channel=chan,
                                                   time=tr.stats.starttime)
                        _ = inv_sel.get_response(tr.id, tr.stats.starttime)

                        tr.remove_response(inventory=inv_sel, output="VEL",
                                           zero_mean=False, taper=False)
                        corrected_channels[chan] = tr.data.astype(np.float32)

                    except Exception as e:
                        write_error(error_path, root, sta_code, chan,
                                    f"Σφάλμα διόρθωσης οργάνου: {e}", net_code)

                if corrected_channels:
                    for chan, corr_data in corrected_channels.items():
                        pdata[chan] = corr_data
                    pdata.save(output_path)
                    print(f"✅ Ολοκληρώθηκε: {output_path}")
                else:
                    print(f"⚠️ Καμία επιτυχής διόρθωση καναλιού: {file}")

            except Exception as e:
                msg = f"❌ Σφάλμα κατά την επεξεργασία {file}: {e}"
                print(msg)
                write_error(error_path, root, "Unknown", "Unknown", msg, "Unknown")


