import os
import numpy as np

def calculate_angles(station_coords, event_coords):
    # Συντεταγμένες: [latitude, longitude, depth]
    station_lat, station_lon, station_depth = station_coords
    event_lat, event_lon, event_depth = event_coords

    # Μετατροπή γεωγραφικού μήκους και πλάτους σε ακτίνια
    lat1, lon1 = np.radians(event_lat), np.radians(event_lon)
    lat2, lon2 = np.radians(station_lat), np.radians(station_lon)

    # Υπολογισμός αζιμουθίου (φι) - απλοποιημένος υπολογισμός
    delta_lon = lon2 - lon1
    x = np.cos(lat2) * np.sin(delta_lon)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    phi = np.arctan2(x, y)

    # Υπολογισμός γωνίας κατάδυσης (θήτα)
    delta_depth = station_depth - event_depth
    distance_horizontal = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    theta = np.arctan2(delta_depth, distance_horizontal)

    # Μετατροπή γωνιών σε μοίρες
    phi = np.degrees(phi)
    theta = np.degrees(theta)

    return phi, theta

def read_msid_file(path):
    return np.fromfile(path, dtype=np.float32)

def write_msid_file(path, data):
    data.astype(np.float32).tofile(path)

def convert_ENZ_files_to_LQT_files(path_E, path_N, path_Z, phi_deg, theta_deg):
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
