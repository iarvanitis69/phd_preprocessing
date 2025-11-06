import numpy as np
from obspy import Trace
from scipy.signal import firwin, filtfilt

import numpy as np

def aic_picker(trace_data):
    """
    Υπολογίζει το AIC σε ολόκληρο το σήμα (μέχρι το pick) και επιστρέφει
    το index όπου ελαχιστοποιείται, ως πιθανή έναρξη του σεισμικού κύματος.

    :param trace_data: numpy array με το σεισμικό σήμα (float, demeaned)
    :return: (index_έναρξης, καμπύλη_AIC)
    """
    data = trace_data.astype(float)
    n = len(data)
    if n < 3:
        return None, np.array([])

    pick_idx = int(np.argmax(np.abs(data)))  # μέγιστη απόλυτη τιμή
    if pick_idx < 10:
        return None, np.array([])  # πολύ μικρό σήμα

    aic = np.zeros(pick_idx)

    for k in range(1, pick_idx - 1):
        var1 = np.var(data[:k]) or 1e-10
        var2 = np.var(data[k:pick_idx]) or 1e-10
        aic[k] = k * np.log(var1) + (pick_idx - k - 1) * np.log(var2)

    min_idx = int(np.argmin(aic[1:pick_idx - 1])) + 1
    return min_idx, aic



def apply_filter_bandpass_fir(trace_data: Trace, freqmin: float, freqmax: float, filter_order: int = 4) -> Trace:
    """
    Εφαρμόζει FIR bandpass φίλτρο στο trace με Butterworth-like απόκριση (σχεδίαση με firwin).

    :param trace_data: Το Obspy Trace object που θα φιλτραριστεί.
    :param freqmin: Κατώτερη συχνότητα του bandpass (Hz).
    :param freqmax: Ανώτερη συχνότητα του bandpass (Hz).
    :param filter_order: Η τάξη του FIR φίλτρου (τυπικά 4 ή 8 για χαμηλή πολυπλοκότητα).
    :return: Ένα νέο Trace object με το φιλτραρισμένο σήμα.
    """
    tr = trace_data.copy()
    fs = tr.stats.sampling_rate  # Συχνότητα δειγματοληψίας

    # Κανονικοποιημένες συχνότητες (0-1) για firwin
    nyq = fs / 2.0
    low = freqmin / nyq
    high = freqmax / nyq

    # Σχεδίαση FIR φίλτρου
    numtaps = filter_order + 1  # Πλήθος coefficients = order + 1
    fir_coeff = firwin(numtaps=numtaps, cutoff=[low, high], pass_zero=False)

    # Εφαρμογή του φίλτρου με zero-phase (μπρος-πίσω)
    tr.data = filtfilt(fir_coeff, [1.0], tr.data)

    return tr

