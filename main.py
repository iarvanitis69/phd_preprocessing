#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import os

from demean_detrend import demean_detrend
from filltering import filter_all_files
from fourier_transformation import find_max_and_min_freq
from gaps import find_files_for_gaps
from glitches import find_files_for_glitches_parallel
from instrument_correction import instrument_correction
from overlaps import find_files_for_overlaps
from peak_segmentation import  find_peak_segmentation
from snr import find_snr
from missingFiles import find_stations_with_issues

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

    #print("🔍 Ξεκινάει ο εντοπισμός στσθμων με λειψα αρχεία...")
    #find_stations_with_issues()

    #print("🔍 Ξεκινάει ο εντοπισμός gaps σε .mseed αρχεία...")
    #find_files_for_gaps()

    #print("🔍 Ξεκινάει ο εντοπισμός overlaps σε .mseed αρχεία...")
    #find_files_for_overlaps()

    #print("🔍 Ξεκινάει ο εντοπισμός glitches σε .mseed αρχεία...")
    #find_files_for_glitches_parallel(threshold=1.3, max_workers=6)

    #print("🔍 κάνει deMean/detrend σε όλα τα mseed αρχεια...")
    #demean_detrend()

    #print("🔍 κάνει instrumentCorrection σε όλα τα mseed αρχεια...")
    #instrument_correction()

    # print("🔍 Ξεκινάει ο υπολογισμός SNR σε *_IC.mseed αρχεία...")
    #find_snr()

    # print("🔍 βρίσκει την μεγιστη συχνοτητα αποκοπής για όλα τα *_demean_detrend.mseed αρχεια...")
    #find_max_and_min_freq()

    # print("🔍 φιλτραρισμα όλων των αρχείων ...")
    #filter_all_files()

    # print("🔍 Βρίσκει το pick segmentation...")
    find_peak_segmentation()

    # print("🔍 Κρατάει μόνο το pick segmentation...")
    #store_peak_segmentation()

    # print("🔍 Conversion to LQT")
    #convert_to_LQT()

    # print("🔍 Normalize Z score all files...")
    #normalize_Z_score()









if __name__ == '__main__':
    multiprocessing.freeze_support()  # προαιρετικό αλλά ασφαλές
    main()

