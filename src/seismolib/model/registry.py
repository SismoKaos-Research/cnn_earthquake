"""Every model in this repo in one table: its knobs, its variants, its inputs.

Not a runnable script -- imported only, and listed by `sk models`.

**The knobs a model has and the flags a task exposes had drifted apart.** The
dual-channel network takes a `branch1d` argument choosing between LSTM-only,
CNN-only and CNN-then-LSTM 1D front ends. Eight detection scripts expose it as
`--branch-1d`; the magnitude regressor, the risk classifier and both forecasters
never did, so for those tasks the CNN front end existed in the class and was
unreachable from the command line. The same drift set `--hidden` to 48 in
detection, 64 in magnitude and forecasting, and 16 in cross-station, each
default living in its own `add_argument` call with nothing relating them.

**A model spec retyped at every evaluation gets retyped wrong.** Seven scripts
carry flags helped `"Must match the checkpoints' training run."` -- the geometry
of a trained model is re-entered by hand each time it is scored, and entering it
wrong builds a different network and loads the checkpoint into it or fails
obscurely. `ModelSpec.save` writes the resolved spec into the checkpoint
directory as `model.json`, and `ModelSpec.load` reads it back, so the record
travels with the weights instead of in a shell history.

Three properties worth stating, because they are what makes this safe to adopt
in a repo whose published numbers came from the old flags:

**The flags generated here are the flags that already exist.** `--channels`,
`--fusion`, `--hidden`, `--fusion-dim`, `--dropout`, `--lstm-layers` and
`--lstm-heads` keep their names, and `--model-branch` accepts `--branch-1d` as
an alias, so every command recorded in a report or a `.sh` runner still runs
verbatim. What is new is `--model`, the alias, and the fact that one table now
answers for all of them.

**Architecture defaults equal the constructor defaults**, and a test asserts it
by reading `inspect.signature` of each registered class. A task that wants
something else passes `defaults=` to `add_model_args`, which is how detection
keeps hidden 48 while magnitude keeps 64 -- visibly, at the call site, instead of
by two unrelated literals.

**Checkpoint filenames are untouched.** `seismolib.checkpoints` matches
`{channels}_{fusion}_{branch_1d}_{seq_transform}_{dataset}_pid_seed` and its
docstring records what adding a flag did to that pattern last time. The spec is
written beside the weights, not into their names.

    from seismolib.model.registry import add_model_args, spec_from_args

    add_model_args(p, family="dual", defaults={"hidden": 48, "fusion_dim": 96,
                                               "dropout": 0.4})
    ...
    spec = spec_from_args(args)
    model = spec.build(seq_dim=seq_shape[-1], img_channels=img_shape[0])
    spec.save(args.save_dir)
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

SPEC_FILENAME = "model.json"


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Param:
    """One tunable of one architecture, and the CLI flag that sets it."""

    name: str
    type: type
    default: object
    help: str
    choices: tuple = None

    @property
    def flag(self):
        """The long option string, e.g. `fusion_dim` -> `--fusion-dim`."""
        return "--" + self.name.replace("_", "-")


@dataclass(frozen=True)
class Architecture:
    """One model: what it is, what it consumes, its variants and its knobs."""

    key: str
    family: str
    summary: str
    inputs: str
    source: str
    build: callable
    params: tuple = ()
    branches: tuple = ()
    branch_help: str = ""
    branch_aliases: tuple = ()
    default_branch: str = None
    notes: str = ""

    def param(self, name):
        """The `Param` with this name, or None."""
        for p in self.params:
            if p.name == name:
                return p
        return None

    def defaults(self):
        """Every param's default, as a dict."""
        return {p.name: p.default for p in self.params}


def _need(shapes, key, arch):
    """Returns `shapes[key]`, or raises naming the architecture that wanted it."""
    if key not in shapes:
        raise ValueError(
            f"model {arch!r} needs shapes[{key!r}]; got {sorted(shapes)}. "
            f"Pass it to ModelSpec.build(), e.g. build({key}=...).")
    return shapes[key]


# ---------------------------------------------------------------------------
# Builders. Imports are deferred so listing the registry costs nothing and
# never pulls a trainer module (and its argparse) in behind a model.
# ---------------------------------------------------------------------------

def _build_dual_channel(branch, p, shapes):
    """DualChannelNet, or DualChannelDualHeadNet when `head="dual"`."""
    from seismolib.model.dual_channel import (DualChannelDualHeadNet,
                                              DualChannelNet)
    common = dict(aux_dim=shapes.get("aux_dim", 0), hidden=p["hidden"],
                  fusion_dim=p["fusion_dim"], dropout=p["dropout"],
                  channels=p["channels"], lstm_layers=p["lstm_layers"],
                  lstm_heads=p["lstm_heads"])
    seq_dim = _need(shapes, "seq_dim", "dual-channel")
    img_channels = _need(shapes, "img_channels", "dual-channel")
    if shapes.get("head") == "dual":
        return DualChannelDualHeadNet(seq_dim, img_channels, fusion=p["fusion"],
                                      branch1d=branch, **common)
    return DualChannelNet(seq_dim, img_channels, fusion=p["fusion"],
                          branch1d=branch,
                          n_classes=shapes.get("n_classes", 1),
                          squeeze_output=shapes.get("squeeze_output", False),
                          **common)


def _build_se_resnet(branch, p, shapes):
    """SETrunk2D; the branch chooses 3 or 4 residual stages."""
    from seismolib.model.trunk2d import SETrunk2D
    return SETrunk2D(num_stages=4 if branch == "res4" else 3,
                     in_channels=shapes.get("in_channels", 3),
                     aux_dim=shapes.get("aux_dim", 0),
                     num_classes=shapes.get("n_classes", 1),
                     dropout1=p["dropout1"], dropout2=p["dropout2"],
                     hidden_dim=p["hidden_dim"])


def _build_sequence_head(branch, p, shapes):
    """SequenceHeadNet; the branch chooses the per-step encoder, if any."""
    from seismolib.model.sequence import SequenceHeadNet
    if branch == "plain":
        return SequenceHeadNet(_need(shapes, "feat_dim", "sequence-head"),
                               hidden=p["hidden"], dropout=p["dropout"])
    if branch == "raw-hour":
        from seismolib.waveform import RawWaveformEncoder
        enc = RawWaveformEncoder(out_dim=p["encoder_dim"], dropout=p["dropout"])
    else:
        from seismolib.waveform import NativeWaveformEncoder
        enc = NativeWaveformEncoder(out_dim=p["encoder_dim"], dropout=p["dropout"])
    return SequenceHeadNet(p["encoder_dim"], hidden=p["hidden"],
                           dropout=p["dropout"], encoder=enc)


def _build_gru(branch, p, shapes):
    """ForecastGRU."""
    from seismolib.model.recurrent import ForecastGRU
    return ForecastGRU(_need(shapes, "feat_dim", "gru"), hidden=p["hidden"],
                       dropout=p["dropout"])


def _build_tcn(branch, p, shapes):
    """ForecastTCN with `levels` blocks of equal width, as every caller used."""
    from seismolib.model.tcn import ForecastTCN
    return ForecastTCN(_need(shapes, "feat_dim", "tcn"),
                       num_channels=[p["hidden"]] * p["levels"],
                       kernel_size=p["kernel_size"], dropout=p["dropout"])


def _build_groundmotion(branch, p, shapes):
    """GroundMotionNet; the branch is its `arch` argument."""
    from groundmotion.cnn_groundmotion import GroundMotionNet
    return GroundMotionNet(arch="cnn_lstm" if branch == "cnn-lstm" else "cnn",
                           n_aux=shapes.get("aux_dim", 0), width=p["width"],
                           hidden=p["hidden"], dropout=p["dropout"],
                           heads=p["heads"])


def _build_cnn_proximity(branch, p, shapes):
    """CNNProximityClassifier."""
    from forecasting.cnn_proximity_classify import CNNProximityClassifier
    return CNNProximityClassifier(cnn_out=p["cnn_out"], dropout=p["dropout"])


def _build_day_3class(branch, p, shapes):
    """DayCNNLSTM3Class."""
    from forecasting.cnn_lstm_daily_3class import DayCNNLSTM3Class
    return DayCNNLSTM3Class(cnn_out=p["cnn_out"], hidden=p["hidden"],
                            dropout=p["dropout"],
                            n_classes=shapes.get("n_classes", 3))


def _build_multiweek(branch, p, shapes):
    """HierarchicalCNNLSTMLSTM."""
    from forecasting.cnn_lstm_lstm_multiweek import HierarchicalCNNLSTMLSTM
    return HierarchicalCNNLSTMLSTM(cnn_out=p["cnn_out"], week_hidden=p["week_hidden"],
                                   seq_hidden=p["seq_hidden"], dropout=p["dropout"],
                                   n_classes=shapes.get("n_classes", 3))


def _build_gru_cnn_fusion(branch, p, shapes):
    """SeismicFusionModel; the branch decides whether the waveform branch exists."""
    from forecasting.gru_cnn import SeismicFusionModel
    return SeismicFusionModel(use_waveform=(branch == "catalog+waveform"),
                              cat_dim=_need(shapes, "cat_dim", "gru-cnn-fusion"),
                              wave_channels=shapes.get("wave_channels", 3),
                              cat_hidden=p["cat_hidden"],
                              wave_embedding=p["wave_embedding"],
                              wave_pool=p["wave_pool"], dropout=p["dropout"])


def _build_catalog_waveform_fusion(branch, p, shapes):
    """CatalogWaveformFusionNet; the branch is its `channels` ablation."""
    from forecasting.cnn_lstm_catalog_waveform_fusion import (CATALOG_DIM,
                                                              CatalogWaveformFusionNet)
    return CatalogWaveformFusionNet(catalog_dim=shapes.get("catalog_dim", CATALOG_DIM),
                                    cnn_out=p["cnn_out"], cat_hidden=p["cat_hidden"],
                                    wave_hidden=p["wave_hidden"],
                                    fusion_hidden=p["fusion_hidden"],
                                    dropout=p["dropout"], channels=branch)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

_DROPOUT = "Dropout used throughout the branches and head."

ARCHITECTURES = (
    Architecture(
        key="dual-channel", family="dual",
        summary="1D waveform branch + 2D image branch (+ optional aux scalars), fused",
        inputs="seq (B,T,C), img (B,C,H,W), aux (B,A) -- the dual-tensor datasets",
        source="seismolib.model.dual_channel:DualChannelNet",
        build=_build_dual_channel,
        branches=("lstm", "cnn", "cnn-lstm"), default_branch="lstm",
        branch_aliases=("--branch-1d",),
        branch_help=(
            "Architecture of the 1D (raw-waveform) branch. lstm: LSTM+attention "
            "over raw samples, which every published result here used. cnn-lstm: "
            "a strided 1D conv encoder first, then that same LSTM+attention -- the "
            "order EQTransformer and PhaseNet use. cnn: conv encoder only, to "
            "isolate whether the recurrence adds anything once local features "
            "exist. No effect under --channels 2d."),
        params=(
            Param("channels", str, "all",
                  "Ablation switch: which branches are active.",
                  choices=("all", "1d", "2d", "aux", "1d+aux", "2d+aux")),
            Param("fusion", str, "linear",
                  "linear: the paper's a*F1+b*F2 (two global scalars). gate: a "
                  "per-example gate g(x)*F1+(1-g(x))*F2. Only affects --channels all.",
                  choices=("linear", "gate")),
            Param("hidden", int, 64, "LSTM hidden size, per direction."),
            Param("fusion_dim", int, 128,
                  "Common width both branches are projected to before fusion."),
            Param("dropout", float, 0.3, _DROPOUT),
            Param("lstm_layers", int, 1, "Stacked LSTM layers in the 1D branch."),
            Param("lstm_heads", int, 4,
                  "Attention heads in the 1D branch. Must divide hidden*2."),
        ),
        notes="shapes head='dual' builds DualChannelDualHeadNet, whose two heads "
              "answer 'will it happen' and 'how big' from one trunk.",
    ),
    Architecture(
        key="se-resnet", family="image",
        summary="SE-ResNet trunk over a RAM image or spectrogram (+ optional aux scalars)",
        inputs="img (B,C,H,W), aux (B,A)",
        source="seismolib.model.trunk2d:SETrunk2D",
        build=_build_se_resnet,
        branches=("res3", "res4"), default_branch="res4",
        branch_help=("Residual stages. res4 (~1.25M params) keeps layer1..layer4 "
                     "as the state-dict keys and matches every existing checkpoint; "
                     "res3 (~0.3M) drops the last stage for short/low-signal inputs."),
        params=(
            Param("dropout1", float, 0.5, "Dropout before the classifier's hidden layer."),
            Param("dropout2", float, 0.3, "Dropout before the classifier's output layer."),
            Param("hidden_dim", int, 64, "Width of the classifier's hidden layer."),
        ),
        notes=("training.ImprovedSeismicCNN subclasses this and is unpickled by "
               "qualified path from full_model.pth, so it stays where it is."),
    ),
    Architecture(
        key="sequence-head", family="sequence",
        summary="BiLSTM+attention over a sequence, with an optional per-step encoder",
        inputs="seq (B,T,F) with branch=plain; (B,T,3,S) raw waveform otherwise",
        source="seismolib.model.sequence:SequenceHeadNet",
        build=_build_sequence_head,
        branches=("plain", "raw-hour", "native-100hz"), default_branch="plain",
        branch_help=("Per-step encoder. plain: none, the step vector goes straight "
                     "into the LSTM (feature_lstm_forecast). raw-hour: "
                     "RawWaveformEncoder over one hour of decimated waveform "
                     "(raw_cnn_lstm_forecast). native-100hz: NativeWaveformEncoder, "
                     "an extra stage and a bigger first stride for 100 Hz input "
                     "(raw100hz_cnn_lstm_forecast)."),
        params=(
            Param("hidden", int, 64, "LSTM hidden size, per direction, and head width."),
            Param("dropout", float, 0.3, _DROPOUT),
            Param("encoder_dim", int, 32,
                  "Per-step embedding width the encoder produces. Ignored by "
                  "--model-branch plain, which has no encoder."),
        ),
    ),
    Architecture(
        key="gru", family="sequence",
        summary="GRU with additive attention pooling, single logit",
        inputs="seq (B,T,F)",
        source="seismolib.model.recurrent:ForecastGRU",
        build=_build_gru,
        params=(
            Param("hidden", int, 64, "GRU hidden size."),
            Param("dropout", float, 0.3, "Dropout in the head (not in the GRU)."),
        ),
    ),
    Architecture(
        key="tcn", family="sequence",
        summary="Dilated causal temporal convolutional network, single logit",
        inputs="seq (B,T,F)",
        source="seismolib.model.tcn:ForecastTCN",
        build=_build_tcn,
        params=(
            Param("hidden", int, 64, "Channels per level."),
            Param("dropout", float, 0.3, _DROPOUT),
            Param("levels", int, 3,
                  "Dilated blocks; dilation doubles at each, so the receptive "
                  "field grows as 2**levels."),
            Param("kernel_size", int, 3, "Convolution kernel width."),
        ),
    ),
    Architecture(
        key="groundmotion", family="window",
        summary="Conv1D trunk over one window, optional BiLSTM+attention, aux head",
        inputs="seq (B,3,300), aux (B,A)",
        source="groundmotion.cnn_groundmotion:GroundMotionNet",
        build=_build_groundmotion,
        branches=("cnn-lstm", "cnn"), default_branch="cnn-lstm",
        branch_aliases=("--arch",),
        branch_help=("cnn-lstm is the paper's stack; cnn pools the trunk directly "
                     "and is the ablation that says whether the recurrent part "
                     "earns its parameters. `cnn_lstm` is accepted as a spelling "
                     "of `cnn-lstm`, matching the old --arch flag."),
        params=(
            Param("width", int, 32, "Base channel width of the Conv1D trunk."),
            Param("hidden", int, 64, "LSTM hidden size, per direction, and head width."),
            Param("dropout", float, 0.2, _DROPOUT),
            Param("heads", int, 4, "Attention heads, when the branch has an LSTM."),
        ),
    ),
    Architecture(
        key="cnn-proximity", family="window",
        summary="RawWaveformEncoder plus a linear head -- no LSTM, no sequence",
        inputs="seq (B,3,S), one hour of raw waveform",
        source="forecasting.cnn_proximity_classify:CNNProximityClassifier",
        build=_build_cnn_proximity,
        params=(
            Param("cnn_out", int, 32, "Width of the CNN's per-hour embedding."),
            Param("dropout", float, 0.3, "Dropout inside the CNN encoder."),
        ),
    ),
    Architecture(
        key="day-3class", family="hierarchical",
        summary="Per-day CNN embedding, then BiLSTM+attention over the days",
        inputs="seq (B, chunk_days, 3, day_samples)",
        source="forecasting.cnn_lstm_daily_3class:DayCNNLSTM3Class",
        build=_build_day_3class,
        params=(
            Param("cnn_out", int, 32, "Width of the CNN's per-day embedding."),
            Param("hidden", int, 16, "LSTM hidden size, per direction, and head width."),
            Param("dropout", float, 0.4, _DROPOUT),
        ),
    ),
    Architecture(
        key="multiweek", family="hierarchical",
        summary="Day CNN, then within-week BiLSTM, then across-weeks BiLSTM",
        inputs="seq (B, n_weeks, 7, 3, day_samples)",
        source="forecasting.cnn_lstm_lstm_multiweek:HierarchicalCNNLSTMLSTM",
        build=_build_multiweek,
        params=(
            Param("cnn_out", int, 32, "Width of the CNN's per-day embedding."),
            Param("week_hidden", int, 16, "Within-week LSTM hidden size, per direction."),
            Param("seq_hidden", int, 16, "Across-weeks LSTM hidden size, per direction."),
            Param("dropout", float, 0.4, _DROPOUT),
        ),
    ),
    Architecture(
        key="gru-cnn-fusion", family="fusion",
        summary="Catalog GRU, optionally fused with a per-hour waveform CNN + GRU",
        inputs="cat_seq (B,T,cat_dim), wave_seq (B,T,3,S)",
        source="forecasting.gru_cnn:SeismicFusionModel",
        build=_build_gru_cnn_fusion,
        branches=("catalog+waveform", "catalog"), default_branch="catalog+waveform",
        branch_help=("catalog builds no waveform branch at all and is the "
                     "catalog-only baseline; catalog+waveform is the full model."),
        params=(
            Param("cat_hidden", int, 32, "Catalog GRU hidden size."),
            Param("wave_embedding", int, 64,
                  "Per-hour waveform embedding width, and the waveform GRU's hidden size."),
            Param("wave_pool", int, 1,
                  "Pooled length per hour before flattening. 1 keeps one vector "
                  "per hour; larger retains coarse within-hour timing."),
            Param("dropout", float, 0.3, "Dropout in the classifier head."),
        ),
    ),
    Architecture(
        key="catalog-waveform-fusion", family="fusion",
        summary="Catalog MLP branch + waveform CNN/BiLSTM branch, concatenated",
        inputs="cat_seq (B,T,catalog_dim), wave_seq (B,T,3,S)",
        source=("forecasting.cnn_lstm_catalog_waveform_fusion:"
                "CatalogWaveformFusionNet"),
        build=_build_catalog_waveform_fusion,
        branches=("all", "catalog", "waveform"), default_branch="all",
        branch_help="Which branch(es) are active; catalog and waveform are the ablations.",
        params=(
            Param("cnn_out", int, 32, "Width of the waveform CNN's per-hour embedding."),
            Param("cat_hidden", int, 16, "Catalog LSTM hidden size, per direction."),
            Param("wave_hidden", int, 16, "Waveform LSTM hidden size, per direction."),
            Param("fusion_hidden", int, 32, "Hidden width of the fusion head."),
            Param("dropout", float, 0.4, _DROPOUT),
        ),
    ),
)

REGISTRY = {a.key: a for a in ARCHITECTURES}

FAMILIES = {
    "dual": "dual-tensor windows (1D waveform + 2D image + aux scalars)",
    "image": "a single 2D image or spectrogram",
    "sequence": "a sequence of per-step feature vectors",
    "window": "one raw waveform window",
    "hierarchical": "raw waveform nested by day and week",
    "fusion": "catalog features fused with raw waveform",
}


def by_family(family=None):
    """The architectures in one family, or all of them, in registration order."""
    if family is None:
        return list(ARCHITECTURES)
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; known: {sorted(FAMILIES)}")
    return [a for a in ARCHITECTURES if a.family == family]


# ---------------------------------------------------------------------------
# The resolved spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """One fully-resolved architecture choice: model, branch, and every param."""

    model: str
    branch: str = None
    params: dict = field(default_factory=dict)

    @property
    def arch(self):
        """The `Architecture` this spec names."""
        return REGISTRY[self.model]

    def build(self, **shapes):
        """Constructs the model. Shape/head arguments come from the dataset.

        Args:
            **shapes: Data-derived values the architecture needs -- `seq_dim`,
                `img_channels`, `aux_dim`, `feat_dim`, `n_classes`,
                `squeeze_output`, `head`. Which ones are required depends on
                the architecture; a missing one raises naming itself.

        Returns:
            An `nn.Module`.
        """
        a = self.arch
        return a.build(self.branch, {**a.defaults(), **self.params}, shapes)

    def to_dict(self):
        """JSON-safe dict of the spec."""
        return {"model": self.model, "branch": self.branch,
                "params": dict(self.params)}

    def save(self, out_dir, filename=SPEC_FILENAME):
        """Writes the spec next to the checkpoints, so evaluation can read it.

        Args:
            out_dir: The `--save-dir` the run writes weights into.
            filename: Name to write; the default is what `load` looks for.

        Returns:
            The `Path` written.
        """
        p = Path(out_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, out_dir, filename=SPEC_FILENAME):
        """Reads a spec written by `save`, or None if the directory has none.

        None rather than an exception: checkpoint directories predating this
        module are the common case, and a caller falling back to its own flags
        is the correct behaviour there, not a crash.
        """
        p = Path(out_dir) / filename
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return cls(model=d["model"], branch=d.get("branch"),
                   params=d.get("params", {}))

    def describe(self):
        """One line naming the model, its branch and its non-default params."""
        a = self.arch
        diff = {k: v for k, v in self.params.items()
                if a.param(k) is not None and v != a.param(k).default}
        bits = [self.model]
        if self.branch:
            bits.append(f"branch={self.branch}")
        bits += [f"{k}={v}" for k, v in sorted(diff.items())]
        return " ".join(bits)


def disagreements(spec, other):
    """Fields where two specs differ, as `{field: (spec_value, other_value)}`.

    Used to warn when the flags an evaluation was given do not match the spec
    saved beside the checkpoints it is about to load -- the failure the seven
    "Must match the checkpoints' training run" flags exist to prevent, and
    cannot.
    """
    out = {}
    if spec.model != other.model:
        out["model"] = (spec.model, other.model)
    if spec.branch != other.branch:
        out["branch"] = (spec.branch, other.branch)
    for k in sorted(set(spec.params) | set(other.params)):
        a, b = spec.params.get(k), other.params.get(k)
        if a != b:
            out[k] = (a, b)
    return out


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def add_model_args(parser, family=None, models=None, default=None,
                   defaults=None, restrict=None, default_branch=None,
                   title="model"):
    """Adds `--model`, `--model-branch` and one flag per param, to `parser`.

    Args:
        parser: The `ArgumentParser` to add to.
        family: Restrict `--model` to one family (see `FAMILIES`). A task
            whose dataset yields dual tensors cannot train a TCN, and offering
            it would be a lie.
        models: Explicit list of keys, instead of a family.
        default: Default `--model`. Defaults to the first offered.
        defaults: Per-param overrides, e.g. `{"hidden": 48}` -- this is how
            detection keeps 48 and magnitude keeps 64, at the call site rather
            than in two unrelated literals.
        restrict: Per-param narrowing of `choices`, e.g.
            `{"channels": ("all", "1d", "2d")}`. The detector's datasets carry
            no auxiliary scalars, so `--channels 1d+aux` there would build the
            same network as `--channels 1d` and quietly record a different
            name for it. A task offers only the values it can honour.
        default_branch: Overrides the architecture's own default branch.
        title: Argument-group title in `--help`.

    Returns:
        The argument group, so a caller can add task-specific model flags to it.

    Raises:
        ValueError: If two offered architectures declare the same param name
            with different types; if `defaults` or `restrict` names a param no
            offered architecture has (which is how a renamed flag would
            otherwise pass silently and take no effect); or if a restriction
            excludes its own default or admits a value the architecture does
            not accept.
    """
    offered = ([REGISTRY[m] for m in models] if models is not None
               else by_family(family))
    if not offered:
        raise ValueError(f"no architectures for family={family!r} models={models!r}")
    overrides = dict(defaults or {})
    narrowed = {k: tuple(v) for k, v in (restrict or {}).items()}

    # One flag can serve several architectures only if they agree on what it
    # means. Types differing is the case that would silently misparse.
    merged = {}
    for a in offered:
        for p in a.params:
            seen = merged.get(p.name)
            if seen is not None and seen.type is not p.type:
                raise ValueError(
                    f"{a.key} and the earlier architecture disagree on the type of "
                    f"{p.flag}: {p.type.__name__} vs {seen.type.__name__}")
            merged.setdefault(p.name, p)

    unknown = (set(overrides) | set(narrowed)) - set(merged)
    if unknown:
        raise ValueError(
            f"defaults/restrict={sorted(unknown)} name no parameter of "
            f"{[a.key for a in offered]}; known: {sorted(merged)}")

    for name, allowed in narrowed.items():
        declared = merged[name].choices
        if declared and not set(allowed) <= set(declared):
            raise ValueError(
                f"restrict[{name!r}]={list(allowed)} admits values the "
                f"architecture does not accept: {sorted(set(allowed) - set(declared))}")
        chosen = overrides.get(name, merged[name].default)
        if chosen not in allowed:
            raise ValueError(
                f"restrict[{name!r}]={list(allowed)} excludes the default "
                f"{chosen!r}; pass a default that survives the restriction")

    g = parser.add_argument_group(title)
    keys = [a.key for a in offered]
    g.add_argument("--model", default=default or keys[0], choices=keys,
                   help="Architecture to build. `sk models` describes each.")

    branch_opts = ["--model-branch"]
    for a in offered:
        branch_opts += [f for f in a.branch_aliases if f not in branch_opts]
    branchy = [a for a in offered if a.branches]
    if branchy:
        # No `choices=` here: which values are legal depends on --model, which
        # argparse cannot express. spec_from_args validates against the model
        # actually chosen and names the legal values for it.
        g.add_argument(*branch_opts, dest="model_branch", default=None,
                       metavar="NAME",
                       help="; ".join(f"{a.key}: {'|'.join(a.branches)}"
                                      for a in branchy) +
                            ". Defaults to " +
                            ", ".join(f"{a.key}={default_branch or a.default_branch}"
                                      for a in branchy) + ".")

    for name, p in merged.items():
        which = [a.key for a in offered if a.param(name) is not None]
        scope = "" if len(which) == len(offered) else f" [{', '.join(which)}]"
        choices = narrowed.get(name, p.choices)
        g.add_argument(p.flag, type=p.type, default=overrides.get(name, p.default),
                       choices=list(choices) if choices else None,
                       help=p.help + scope)

    parser.set_defaults(_model_offered=keys, _model_default_branch=default_branch)
    return g


def spec_from_args(args):
    """Builds a `ModelSpec` from args parsed by a parser `add_model_args` touched.

    Args:
        args: The parsed namespace.

    Returns:
        A `ModelSpec` holding the chosen model, its resolved branch, and only
        the params that model actually has -- so a spec never records a flag
        the architecture ignores.

    Raises:
        ValueError: If `--model-branch` is not a branch of the chosen model.
    """
    a = REGISTRY[args.model]
    branch = getattr(args, "model_branch", None)
    if branch is None:
        branch = getattr(args, "_model_default_branch", None) or a.default_branch
    if a.branches:
        branch = _canonical_branch(a, branch)
        if branch not in a.branches:
            raise ValueError(
                f"--model-branch {branch!r} is not a branch of --model {a.key}; "
                f"choose one of {', '.join(a.branches)}")
    elif branch is not None and branch not in (None, "default"):
        raise ValueError(f"--model {a.key} has no branches, but --model-branch "
                         f"{branch!r} was given")
    else:
        branch = None
    params = {p.name: getattr(args, p.name) for p in a.params
              if hasattr(args, p.name)}
    return ModelSpec(model=a.key, branch=branch, params=params)


def _canonical_branch(arch, branch):
    """Accepts an underscored spelling of a hyphenated branch, e.g. `cnn_lstm`."""
    if branch is not None and branch not in arch.branches:
        alt = branch.replace("_", "-")
        if alt in arch.branches:
            return alt
    return branch
