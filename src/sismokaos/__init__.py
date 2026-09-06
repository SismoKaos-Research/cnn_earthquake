"""Shared library for the seismic model scripts.

Everything here is imported, never run. The rule for what belongs: if two
scripts in different families would otherwise each carry a copy, it goes here.

    metrics    accuracy/AUC reports and trivial baselines
    training   seeding, training loops, early stopping
    catalog    event-catalog loading, hourly labelling, distance helpers
    splits     walk-forward chronological CV and its diagnostics
    waveform   hourly raw-waveform loading and the CNN encoder
    baselines  conditional floors (persistence, majority, amplitude)
    logging    DualLogger
    model/     network branches, trunks and fusion
    data/      torch Dataset wrappers
"""
