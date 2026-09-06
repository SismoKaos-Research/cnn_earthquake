"""What this repo can be trained to predict, and which model answers each.

Not a runnable script -- imported only, and printed by `sk train`.

The trainers are named after their architecture (`cnn_lstm_classify`,
`raw_cnn_lstm_forecast`, `feature_gru_tcn`), which is the wrong axis to look
along. Two scripts whose names differ by one word learn completely different
labels, and two that learn the same label sit in different packages. The
question anyone actually starts from is "what am I predicting" -- is there an
event in this window, how big is it, is one coming, how hard will it shake --
and the answer to that was not written down anywhere.

This is that index: task -> the label it learns -> the module that trains it.
`seismolib.model.registry` answers the other half, which architectures a task
may use, and the two meet at `sk train <task> --model <name>`.

**A task's `family` is what makes `--model` real.** `forecast-features` names
the sequence family, so all three of `sequence-head`, `gru` and `tcn` train
against its labels and its splits -- previously that comparison meant a second
script with its own split logic. A task with no family dispatches to its trainer
unchanged and takes whatever flags that trainer already had; nothing here
rewrites a trainer's arguments.
"""
from dataclasses import dataclass

# What the label answers. The grouping the tasks are listed under.
PREDICTS = {
    "detect": "is there an earthquake in this window",
    "magnitude": "how big the earthquake in this window is",
    "forecast": "whether an earthquake is coming, and when",
    "shaking": "how hard the ground will move",
}


@dataclass(frozen=True)
class Task:
    """One trainable target: its label, its trainer, and its model choices."""

    key: str
    predicts: str
    module: str
    summary: str
    label: str
    family: str = None
    models: tuple = ()

    @property
    def choosable(self):
        """True when `--model` selects among registered architectures."""
        return bool(self.family or self.models)


TASKS = {t.key: t for t in (
    # ---- is there an earthquake in this window ----------------------------
    Task("detect", "detect", "detection.cnn_lstm_classify",
         "dual-channel event/noise classifier on short arrival-anchored windows",
         "a window is positive if it is anchored on a catalogued arrival, "
         "negative if it is quiet record from the same stations",
         family="dual"),
    Task("detect-aux", "detect", "detection.cnn_lstm_classify_aux",
         "the same classifier with auxiliary scalars concatenated after fusion",
         "as `detect`, plus per-window scalars (distance, depth, SNR)",
         family="dual"),
    Task("detect-spec", "detect", "detection.cnn_from_tensor",
         "SE-ResNet over spectrogram tensors -- the 2D-only ancestor",
         "as `detect`, but the window is presented only as a spectrogram"),
    Task("detect-png", "detect", "detection.cnn_train",
         "SE-ResNet over RAM images read through ImageFolder",
         "as `detect-spec`, from PNG recurrence plots on disk"),
    Task("detect-ram-aux", "detect", "detection.cnn_ram_aux",
         "RAM-image SE-ResNet with auxiliary scalars",
         "as `detect-png`, plus per-window scalars"),
    Task("detect-cross-station", "detect", "detection.cnn_lstm_cross_station",
         "trains on one station's windows and tests on another's",
         "as `detect`, with the split forced across stations rather than events"),

    # ---- how big ----------------------------------------------------------
    Task("magnitude", "magnitude", "magnitude.cnn_lstm_regression",
         "dual-channel magnitude regression",
         "the catalogue magnitude of the event the window is anchored on",
         family="dual"),
    Task("magnitude-image", "magnitude", "magnitude.cnn_regression",
         "SE-ResNet magnitude regression from the 2D representation alone",
         "as `magnitude`, from the image channel only"),
    Task("magclass", "magnitude", "magnitude.cnn_magclass",
         "magnitude as ordered bands rather than a scalar",
         "the magnitude bin the event falls in"),
    Task("riskclass", "magnitude", "magnitude.cnn_riskclass",
         "risk class from a single window",
         "a risk band derived from magnitude and distance together"),

    # ---- is one coming ----------------------------------------------------
    Task("forecast", "forecast", "forecasting.cnn_lstm_forecast",
         "dual-channel forecaster with two heads: will it happen, and how big",
         "a catalogue window is positive if an event over --threshold occurs "
         "within --horizon-days of its end",
         family="dual"),
    Task("risk", "forecast", "forecasting.cnn_lstm",
         "three-way risk class from a catalogue window",
         "time to the next mainshock, bucketed into ordered classes",
         family="dual"),
    Task("forecast-features", "forecast", "forecasting.feature_lstm_forecast",
         "forecasting from hand-crafted hourly features",
         "an hour is positive if an event over --threshold occurs within "
         "--horizon-days of it",
         family="sequence"),
    Task("forecast-gru-tcn", "forecast", "forecasting.feature_gru_tcn",
         "GRU and TCN on the same hourly features, trained and compared together",
         "as `forecast-features`; this one trains both architectures in one run "
         "rather than selecting between them"),
    Task("forecast-raw", "forecast", "forecasting.raw_cnn_lstm_forecast",
         "per-hour waveform CNN, then LSTM+attention over the hours",
         "as `forecast-features`, but from raw waveform instead of features"),
    Task("forecast-chunk", "forecast", "forecasting.cnn_chunk_forecast",
         "forecasting from multi-day raw chunks",
         "a chunk is positive if a qualifying event follows its end"),
    Task("daily-3class", "forecast", "forecasting.cnn_lstm_daily_3class",
         "day embeddings, then LSTM+attention, into three regime classes",
         "before / during / after a qualifying event"),
    Task("multiweek", "forecast", "forecasting.cnn_lstm_lstm_multiweek",
         "day -> week -> multi-week hierarchy into three regime classes",
         "as `daily-3class`, over several weeks of context"),
    Task("proximity", "forecast", "forecasting.cnn_proximity_classify",
         "single-hour CNN with no sequence at all -- the no-LSTM control",
         "an hour is positive if a qualifying event is within --close-days "
         "in either direction"),
    Task("catalog-waveform", "forecast", "forecasting.cnn_lstm_catalog_waveform_fusion",
         "catalogue features and raw waveform fused, trained end to end",
         "as `forecast-features`, with both inputs available at once"),
    Task("gru-cnn", "forecast", "forecasting.gru_cnn_train",
         "catalogue GRU with an optional per-hour waveform branch",
         "as `forecast-features`, with the waveform branch ablatable"),
    Task("next-event", "forecast", "forecasting.next_event_regression",
         "how long until the next qualifying event, as a regression",
         "days from the window's end to the next event over --threshold"),
    Task("loeo", "forecast", "forecasting.cnn_lstm_loeo",
         "leave-one-event-out evaluation of the risk model",
         "as `risk`, with every event held out in turn"),
    Task("chaos", "forecast", "forecasting.chaos_forecast",
         "forecasting from nonlinear-dynamics features",
         "as `forecast-features`, from chaos features instead"),

    # ---- how hard will it shake -------------------------------------------
    Task("groundmotion", "shaking", "groundmotion.cnn_groundmotion",
         "Conv1D trunk, optional BiLSTM+attention, predicting PGA/PGV",
         "the peak ground acceleration or velocity measured on the window",
         models=("groundmotion",)),
)}


def by_prediction(predicts=None):
    """Tasks under one prediction group, or all of them, in listed order."""
    if predicts is not None and predicts not in PREDICTS:
        raise ValueError(f"unknown group {predicts!r}; known: {sorted(PREDICTS)}")
    return [t for t in TASKS.values()
            if predicts is None or t.predicts == predicts]
