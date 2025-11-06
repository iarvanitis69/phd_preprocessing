#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import os

from demean_detrend import demean_detrend
from fourier_transformation import find_max_and_min_freq
from gaps import find_files_for_gaps
from glitches import find_files_for_glitches_parallel, delete_files_with_glitches
from instrument_correction import instrument_correction
from overlaps import find_files_for_overlaps
from peak_segmentation import  find_start_end_and_peak_of_signal
from snr import find_snr
from stationsWith3Channels import find_stations_with_nofChannelsL3, delete_stations_with_nofChannels_l3

# === ΡΥΘΜΙΣΕΙΣ ===
BASE_DIR = "/media/iarv/Samsung/Events"  # 👉 άλλαξέ το αν χρειάζεται
LOG_DIR = os.path.join(BASE_DIR, "Logs")
os.makedirs(LOG_DIR, exist_ok=True)
OVERLAPS_LOG_FILE = os.path.join(LOG_DIR, "overlaps.json")
GAPS_FILE = os.path.join(LOG_DIR, "gaps.json")  # ✅ αρχείο εξόδου
LOG_FILE = os.path.join(LOG_DIR, "missing_mseed_files.log")


def main():
    # Δημιούργησε εδώ ό,τι χρειάζεσαι
    lock = multiprocessing.Manager().Lock()

    # Αν θες μόνο καταγραφή:
    #find_stations_with_nofChannelsL3()

    # Αν θες διαγραφή:
    #delete_stations_with_nofChannels_l3()

    #print("🔍 Ξεκινάει ο εντοπισμός gaps σε .mseed αρχεία...")
    #find_files_for_gaps()

    #print("🔍 Ξεκινάει ο εντοπισμός overlaps σε .mseed αρχεία...")
    #find_files_for_overlaps()

    #print("🔍 Ξεκινάει ο εντοπισμός glitches σε .mseed αρχεία...")
    #find_files_for_glitches_parallel(threshold=1.3, max_workers=6)

    # print("🔍 Ξεκινάει η διαγραφή stations με glitches σε .mseed αρχεία...")
    #delete_files_with_glitches()

    # print("🔍 κάνει deMean/detrend σε όλα τα mseed αρχεια...")
    #demean_detrend()

    # print("🔍 κάνει instrumentCorrection σε όλα τα mseed αρχεια...")
    #instrument_correction()

    # print("🔍 Ξεκινάει ο υπολογισμός SNR σε *_IC.mseed αρχεία...")
    #find_snr()

    # print("🔍 Διαβαζει όλα τα stations από το snrl55.json και τα σβήνει...")
    #delete_stations_with_snr_lt5()

    # print("🔍 βρίσκει την μεγιστη συχνοτητα αποκοπής για όλα τα *_demean_detrend.mseed αρχεια...")
    find_max_and_min_freq()

    # print("🔍 φιλτραρισμα όλων των αρχείων ...")
    #filter_all_files()()

    # print("🔍 Βρίσκει το pick segmentation...")
    #find_start_end_and_peak_of_signal()

    # print("🔍 Κρατάει μόνο το pick segmentation...")
    #store_peak_segmentation()







if __name__ == '__main__':
    multiprocessing.freeze_support()  # προαιρετικό αλλά ασφαλές
    main()

