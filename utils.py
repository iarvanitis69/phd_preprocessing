import numpy as np
from obspy import Trace
from scipy.signal import firwin, filtfilt

def aic_picker(trace_data):
    """
    Επιστρέφει το index όπου το AIC ελαχιστοποιείται (πιθανή αρχή κύματος).

    :param trace_data: numpy array με το αποσυνετραρισμένο σεισμικό σήμα
    :return: index του πιθανού σημείου έναρξης κύματος
    """
    data = trace_data - np.mean(trace_data)
    n = len(data)
    aic = np.zeros(n)

    for k in range(1, n - 1):
        var1 = np.var(data[:k]) if k > 1 else 1e-10
        var2 = np.var(data[k:]) if k < n - 2 else 1e-10
        aic[k] = k * np.log(var1) + (n - k - 1) * np.log(var2)

    return np.argmin(aic), aic

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

