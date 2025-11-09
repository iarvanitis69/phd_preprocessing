import os
import numpy as np
from obspy.geodetics.base import gps2dist_azimuth, locations2degrees
import json

def convert_ENZ_to_LQT_files(base_dir, excluded_stations, event_coords):
    """
    Αναζητά αναδρομικά όλα τα ENZ αρχεία ανά σταθμό και δημιουργεί LQT αρχεία,
    εξαιρώντας όποιους σταθμούς δίνονται στο excluded_stations.

    Parameters:
        base_dir: Διαδρομή προς τον βασικό φάκελο με τα γεγονότα
        excluded_stations: Λίστα με ονόματα σταθμών προς παράλειψη (π.χ. ["HL.SANT", "HT.APE"])
        event_coords: [lat, lon, depth] του σεισμού σε km

    Δημιουργεί αρχεία *_L.mseed, *_Q.mseed, *_T.mseed σε κάθε φάκελο σταθμού
    """
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_E.msid"):
                # path_E = os.path.join(root, file)
                # path_N = path_E.replace("_E.msid", "_N.msid")
                # path_Z = path_E.replace("_E.msid", "_Z.msid")
                #
                # if not (os.path.exists(path_N) and os.path.exists(path_Z)):
                #     print(f"❌ Λείπει κάποιο από τα N ή Z για: {file}")
                #     continue

                # Από το path, απομονώνουμε το σταθμό (π.χ. HL.SANT)
                parts = os.path.normpath(path_E).split(os.sep)
                if len(parts) < 2:
                    continue
                station = parts[-2]  # Αναμένουμε δομή: .../<event>/<station>/...

                if station in excluded_stations:
                    print(f"⏭️ Παράλειψη σταθμού: {station}")
                    continue

                # Ανάκτηση συντεταγμένων σταθμού από info.json
                info_path = os.path.join(root, "..", "..", "Stations", station, "info.json")
                info_path = os.path.abspath(info_path)
                if not os.path.exists(info_path):
                    print(f"❌ Δεν βρέθηκε info.json για {station}")
                    continue

                with open(info_path) as f:
                    station_info = json.load(f)
                station_coords = [
                    station_info["latitude"],
                    station_info["longitude"],
                    station_info.get("elevation_km", 0.0)
                ]

                try:
                    phi, theta = calculate_angles(station_coords, event_coords)
                    convert_ENZ_files_to_LQT_for_station(path_E, path_N, path_Z, phi, theta)
                    print(f"✅ Μετατροπή LQT για {station} στο {root}")
                except Exception as e:
                    print(f"❌ Σφάλμα σε {station}: {e}")

def calculate_angles(station_coords, event_coords):
    """
    Υπολογίζει την αζιμουθιακή γωνία (phi) και τη γωνία κατάδυσης (theta)
    από τον σεισμό προς τον σταθμό, χρησιμοποιώντας την obspy.geodetics.base.gps2dist_azimuth.

    Parameters:
        station_coords: [lat, lon, depth] σε km
        event_coords: [lat, lon, depth] σε km

    Returns:
        phi_deg: αζιμουθιακή γωνία σε μοίρες [0–360]
        theta_deg: γωνία κατάδυσης σε μοίρες
    """
    station_lat, station_lon, station_depth = station_coords
    event_lat, event_lon, event_depth = event_coords

    # Υπολογισμός αζιμουθιακής γωνίας με ObsPy
    _, azimuth_deg, _ = gps2dist_azimuth(event_lat, event_lon, station_lat, station_lon)
    phi_deg = azimuth_deg  # ήδη σε μοίρες

    # Υπολογισμός επιφανειακής απόστασης σε km (με great-circle distance)
    deg_distance = locations2degrees(event_lat, event_lon, station_lat, station_lon)
    R = 6371.0  # Ακτίνα Γης σε km
    horizontal_distance_km = np.radians(deg_distance) * R

    # Υπολογισμός γωνίας κατάδυσης (theta)
    delta_depth_km = station_depth - event_depth
    theta_rad = np.arctan2(delta_depth_km, horizontal_distance_km)
    theta_deg = np.degrees(theta_rad)

    return phi_deg, theta_deg


def read_msid_file(path):
    return np.fromfile(path, dtype=np.float32)

def write_msid_file(path, data):
    data.astype(np.float32).tofile(path)

def convert_ENZ_files_to_LQT_for_station(path_E, path_N, path_Z, phi_deg, theta_deg):
    # Ανάγνωση δεδομένων
    E = read_msid_file(path_E)
    N = read_msid_file(path_N)
    Z = read_msid_file(path_Z)

    if not (len(E) == len(N) == len(Z)):
        raise ValueError("Τα αρχεία E, N, Z έχουν διαφορετικό μήκος!")

    # Μετατροπή γωνιών σε ακτίνια
    phi = np.radians(phi_deg)
    theta = np.radians(theta_deg)

    # Μετασχηματισμός ENZ → LQT
    L = np.sin(theta) * np.cos(phi) * E + np.sin(theta) * np.sin(phi) * N + np.cos(theta) * Z
    Q = np.cos(theta) * np.cos(phi) * E + np.cos(theta) * np.sin(phi) * N - np.sin(theta) * Z
    T = -np.sin(phi) * E + np.cos(phi) * N

    # Δημιουργία paths εξόδου με βάση το path_E
    base_folder = os.path.dirname(path_E)
    base_name = os.path.basename(path_E)

    # Αφαίρεση suffix (E.msid) και αντικατάσταση
    if "_E.msid" in base_name:
        base_prefix = base_name.replace("_E.mseed", "")
    else:
        raise ValueError("Το όνομα του αρχείου E δεν είναι της μορφής *_E.mseed")

    path_L = os.path.join(base_folder, f"{base_prefix}_L.mseed")
    path_Q = os.path.join(base_folder, f"{base_prefix}_Q.mseed")
    path_T = os.path.join(base_folder, f"{base_prefix}_T.mseed")

    # Εγγραφή αρχείων
    write_msid_file(path_L, L)
    write_msid_file(path_Q, Q)
    write_msid_file(path_T, T)

    return path_L, path_Q, path_T
