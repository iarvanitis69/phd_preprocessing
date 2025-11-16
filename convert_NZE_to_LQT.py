import os
import numpy as np
from obspy.geodetics.base import gps2dist_azimuth, locations2degrees
import json

def convert_NZE_to_LQT():
    """
    Converts NZE (HHN, HHE, HHZ or BHN, BHE, BHZ) signals into LQT signals
    for ALL three sets:
        • PSfixed
        • PSvariant
        • WholeEvent

    For κάθε set (π.χ. PSfixed):
        - Εντοπίζει 3 αρχεία (N,E,Z)
        - Υπολογίζει LQT rotation χρησιμοποιώντας το επίκεντρο από info.json
          και τη θέση του σταθμού από το network.station.xml
        - Παράγει 3 νέα αρχεία:
              *_L.mseed
              *_Q.mseed
              *_T.mseed

    Σύνολο παραγόμενων αρχείων: 9 ανά σταθμό (3 sets × 3 components).
    """

    import os
    import numpy as np
    from obspy import read
    from obspy.geodetics import gps2dist_azimuth

    from main import BASE_DIR
    from utils import load_json

    # ===============================================
    # Βοηθητικό: υπολογισμός rotation matrix R (3x3)
    # ===============================================
    def rotation_matrix_LQT(stla, stlo, evla, evlo):
        """
        Υπολογίζει τον πίνακα LQT rotation:
        L: άξονας προς το επίκεντρο
        Q: οριζόντια κάθετη συνιστώσα
        T: κατακόρυφη συνιστώσα

        Επιστρέφει: 3×3 rotation matrix
        """

        # Διεύθυνση από σταθμό → επίκεντρο
        _, az, _ = gps2dist_azimuth(stla, stlo, evla, evlo)
        baz = (az + 180) % 360

        # Μετατροπή μοίρες → rad
        az_r = np.deg2rad(az)
        baz_r = np.deg2rad(baz)

        # L: radial direction
        L = np.array([
            np.sin(baz_r),      # North
            np.cos(baz_r),      # East
            0
        ])

        # Q: transverse (left)
        Q = np.array([
            np.cos(baz_r),
            -np.sin(baz_r),
            0
        ])

        # T: vertical
        T = np.array([0, 0, 1])

        R = np.vstack([L, Q, T])
        return R

    # ===============================================
    # Σάρωση όλων των ετών
    # ===============================================
    for year in os.listdir(BASE_DIR):
        year_path = os.path.join(BASE_DIR, year)
        if not os.path.isdir(year_path):
            continue

        # Σάρωση events
        for event_name in os.listdir(year_path):

            event_path = os.path.join(year_path, event_name)
            if not os.path.isdir(event_path):
                continue

            # ------------- Load epicenter info.json -------------
            info_path = os.path.join(event_path, "info.json")
            if not os.path.exists(info_path):
                continue

            info = load_json(info_path)
            evla = float(info["latitude"])
            evlo = float(info["longitude"])

            # ----------- Σάρωση σταθμών -------------------
            for station_name in os.listdir(event_path):

                station_path = os.path.join(event_path, station_name)
                if not os.path.isdir(station_path):
                    continue
                if station_name == "info.json":
                    continue

                # -------- Load station coordinates from XML --------
                xml_path = os.path.join(station_path, f"{station_name}.xml")
                if not os.path.exists(xml_path):
                    print(f"⚠️ Missing XML for {station_name}")
                    continue

                try:
                    from obspy import read_inventory
                    inv = read_inventory(xml_path)
                    net = inv.networks[0]
                    sta = net.stations[0]
                    stla = sta.latitude
                    stlo = sta.longitude
                except Exception as e:
                    print(f"⚠️ Cannot read XML {xml_path}: {e}")
                    continue

                # -------- Rotation matrix --------
                R = rotation_matrix_LQT(stla, stlo, evla, evlo)

                # ======================================================
                #  Επεξεργασία 3 SETS:
                #      - PSfixed
                #      - PSvariant
                #      - WholeEvent
                # ======================================================
                waveform_sets = {
                    "PSfixed":    "_demeanDetrend_IC_BPF_PSfixed.mseed",
                    "PSvariant":  "_demeanDetrend_IC_BPF_PSvariant.mseed",
                    "WholeEvent": "_demeanDetrend_IC_BPF_WholeEvent.mseed"
                }

                for set_name, suffix in waveform_sets.items():

                    # --- Εντοπισμός αρχείων N,E,Z ---
                    chan_files = {"N": None, "E": None, "Z": None}

                    for f in os.listdir(station_path):
                        if not f.endswith(suffix):
                            continue

                        full = os.path.join(station_path, f)

                        if "__HHN" in f:
                            chan_files["N"] = full
                        elif "__HHE" in f:
                            chan_files["E"] = full
                        elif "__HHZ" in f:
                            chan_files["Z"] = full

                    # Αν λείπει κανάλι → skip
                    if None in chan_files.values():
                        continue

                    # ---- Load N/E/Z traces ----
                    try:
                        trN = read(chan_files["N"])[0]
                        trE = read(chan_files["E"])[0]
                        trZ = read(chan_files["Z"])[0]
                    except Exception as e:
                        print(f"⚠️ Failed reading NZE for {station_name}: {e}")
                        continue

                    # ---- Check same length ----
                    if not (len(trN.data) == len(trE.data) == len(trZ.data)):
                        print(f"⚠️ Mismatch lengths for {station_name} {set_name}")
                        continue

                    # ---- Construct 3×N matrix ----
                    M = np.vstack([
                        trN.data.astype(np.float32),
                        trE.data.astype(np.float32),
                        trZ.data.astype(np.float32)
                    ])

                    # ---- Apply rotation ----
                    LQT = R @ M

                    # ---- Write L / Q / T ----
                    components = ["L", "Q", "T"]

                    for i, comp in enumerate(components):

                        tr_new = trZ.copy()
                        tr_new.data = LQT[i].astype(np.float32)

                        outname = chan_files["Z"].replace(".mseed", f"_{comp}.mseed")
                        tr_new.write(outname, format="MSEED")

                        print(f"✔ Created {set_name} → {outname}")


# ==========================================================
if __name__ == "__main__":
    convert_NZE_to_LQT()
