"""Every registered architecture builds, runs, and agrees with its own class.

The registry exists to stop the model's knobs and the task's flags from
drifting apart (`sismokaos/model/registry.py` explains what that drift cost).
A registry that itself drifts from the classes it describes would be worse than
no registry, so the checks here are deliberately about agreement with the code
rather than about the registry's internal consistency:

**Every architecture is constructed and run**, on a batch of the shape its
`inputs` field advertises, for every branch it declares. A builder that passes
a keyword the class does not take, or a branch string the class rejects, fails
here rather than forty minutes into a training run.

**Registry defaults equal constructor defaults.** A `Param` whose name matches a
constructor argument must carry that argument's default, so the registry cannot
quietly change a model's shape. Params with no matching argument -- `tcn`'s
`levels`, which the registry synthesises into `num_channels` -- are skipped, and
the test reports how many it checked so the skipping cannot hollow it out.

**A registry-built dual-channel model is state-dict-identical to the
hand-constructed `DualChannelBinaryNet`.** That is the compatibility claim that
matters: existing detector checkpoints must load into a model the registry
built, or converting the trainer would strand every result in the repo.
"""
import argparse
import inspect
import json
import sys

import pytest
import torch

from sismokaos.model.registry import (ARCHITECTURES, FAMILIES, REGISTRY,
                                      ModelSpec, add_model_args, by_family,
                                      disagreements, spec_from_args)

# Per architecture: the shapes its build() needs, and a batch to run through it.
# Sizes are the smallest that survive each model's stride stack, not the real
# ones -- this asserts the wiring, not the science.
HOUR = 4096          # long enough for RawWaveformEncoder's 4 stride-4 blocks
NATIVE = 9600        # NativeWaveformEncoder strides multiply to 4800

CASES = {
    "dual-channel": (dict(seq_dim=3, img_channels=3),
                     lambda: (torch.randn(2, 600, 3), torch.randn(2, 3, 32, 32)),
                     (2, 1)),
    "se-resnet": (dict(in_channels=3),
                  lambda: (torch.randn(2, 3, 32, 32),), (2, 1)),
    "sequence-head": (dict(feat_dim=11),
                      lambda: (torch.randn(2, 20, 11),), (2,)),
    "gru": (dict(feat_dim=11), lambda: (torch.randn(2, 20, 11),), (2,)),
    "tcn": (dict(feat_dim=11), lambda: (torch.randn(2, 20, 11),), (2,)),
    "groundmotion": (dict(aux_dim=2),
                     lambda: (torch.randn(2, 3, 300), torch.randn(2, 2)), (2,)),
    "cnn-proximity": (dict(), lambda: (torch.randn(2, 3, HOUR),), (2,)),
    "day-3class": (dict(), lambda: (torch.randn(2, 4, 3, HOUR),), (2, 3)),
    "multiweek": (dict(), lambda: (torch.randn(2, 2, 7, 3, HOUR),), (2, 3)),
    "gru-cnn-fusion": (dict(cat_dim=3),
                       lambda: (torch.randn(2, 6, 3), torch.randn(2, 6, 3, HOUR)),
                       (2, 1)),
    "catalog-waveform-fusion": (dict(catalog_dim=13),
                                lambda: (torch.randn(2, 6, 3, HOUR),
                                         torch.randn(2, 6, 13)), (2,)),
}

# Branches whose forward takes a different input than the architecture's
# default case above.
BRANCH_INPUTS = {
    ("sequence-head", "raw-hour"): lambda: (torch.randn(2, 6, 3, HOUR),),
    ("sequence-head", "native-100hz"): lambda: (torch.randn(2, 3, 3, NATIVE),),
    ("gru-cnn-fusion", "catalog"): lambda: (torch.randn(2, 6, 3),),
}


def test_every_architecture_has_a_case():
    """A new architecture without a case here would go untested silently."""
    assert set(CASES) == set(REGISTRY), (
        f"CASES and the registry disagree: {set(CASES) ^ set(REGISTRY)}")


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_builds_and_runs(key):
    """Default branch, default params: constructs and produces the right shape."""
    shapes, inputs, out_shape = CASES[key]
    model = ModelSpec(model=key, branch=REGISTRY[key].default_branch).build(**shapes)
    model.eval()
    with torch.no_grad():
        out = model(*inputs())
    assert tuple(out.shape) == out_shape, f"{key} returned {tuple(out.shape)}"


@pytest.mark.parametrize("key,branch",
                         [(a.key, b) for a in ARCHITECTURES for b in a.branches])
def test_every_branch_builds_and_runs(key, branch):
    """Each declared branch is a value the underlying class actually accepts."""
    shapes, inputs, out_shape = CASES[key]
    inputs = BRANCH_INPUTS.get((key, branch), inputs)
    model = ModelSpec(model=key, branch=branch).build(**shapes)
    model.eval()
    with torch.no_grad():
        out = model(*inputs())
    assert tuple(out.shape) == out_shape, f"{key}/{branch} returned {tuple(out.shape)}"


def test_registry_defaults_match_constructor_defaults():
    """A Param that names a constructor argument must carry its default."""
    checked = 0
    for a in ARCHITECTURES:
        module, cls_name = a.source.split(":")
        cls = getattr(__import__(module, fromlist=[cls_name]), cls_name)
        sig = inspect.signature(cls.__init__).parameters
        for p in a.params:
            if p.name not in sig or sig[p.name].default is inspect.Parameter.empty:
                continue           # synthesised, e.g. tcn's levels -> num_channels
            checked += 1
            assert p.default == sig[p.name].default, (
                f"{a.key}.{p.name} defaults to {p.default!r} in the registry but "
                f"{sig[p.name].default!r} in {a.source} -- the registry would "
                f"silently change the model's shape")
    # Guard against the loop skipping everything if `source` strings go stale.
    assert checked >= 25, f"only {checked} defaults were actually compared"


def test_dual_channel_matches_the_hand_built_detector():
    """Registry-built == DualChannelBinaryNet, so existing checkpoints load."""
    from sismokaos.detection.cnn_lstm_classify import DualChannelBinaryNet

    kw = dict(hidden=48, fusion_dim=96, dropout=0.4, channels="all",
              fusion="linear", branch1d="cnn-lstm")
    torch.manual_seed(0)
    hand = DualChannelBinaryNet(3, 3, **kw)
    torch.manual_seed(0)
    built = ModelSpec(model="dual-channel", branch="cnn-lstm",
                      params={k: v for k, v in kw.items() if k != "branch1d"}
                      | {"lstm_layers": 1, "lstm_heads": 4}).build(
        seq_dim=3, img_channels=3)

    assert list(hand.state_dict()) == list(built.state_dict())
    assert (sum(p.numel() for p in hand.parameters())
            == sum(p.numel() for p in built.parameters()))
    built.load_state_dict(hand.state_dict())          # the claim that matters
    hand.eval(), built.eval()
    seq, img = torch.randn(2, 600, 3), torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(hand(seq, img), built(seq, img))


@pytest.mark.parametrize("branch", ["lstm", "cnn", "cnn-lstm"])
@pytest.mark.parametrize("fusion", ["linear", "gate"])
def test_dual_head_variant_takes_every_trunk_option(branch, fusion):
    """Two heads constrain nothing about the trunk they share."""
    model = ModelSpec(model="dual-channel", branch=branch,
                      params={"fusion": fusion}).build(
        seq_dim=3, img_channels=3, head="dual")
    model.eval()
    with torch.no_grad():
        binary, mag = model(torch.randn(2, 600, 3), torch.randn(2, 3, 32, 32))
    assert binary.shape == mag.shape == (2,)


def test_missing_shape_names_itself():
    """A forgotten shape says which model wanted which key, not KeyError."""
    with pytest.raises(ValueError, match="dual-channel.*img_channels"):
        ModelSpec(model="dual-channel", branch="lstm").build(seq_dim=3)


# ---------------------------------------------------------------------------
# The CLI surface
# ---------------------------------------------------------------------------

def _parser(**kw):
    p = argparse.ArgumentParser()
    add_model_args(p, **kw)
    return p


def test_flags_keep_the_names_existing_commands_use():
    """Every recorded `--channels/--fusion/--branch-1d ...` command still parses."""
    args = _parser(family="dual", defaults={"hidden": 48, "fusion_dim": 96,
                                            "dropout": 0.4}).parse_args(
        ["--channels", "all", "--fusion", "gate", "--branch-1d", "cnn-lstm"])
    spec = spec_from_args(args)
    assert spec.branch == "cnn-lstm"
    assert spec.params["fusion"] == "gate"
    assert spec.params["hidden"] == 48, "the task's default override was lost"


def test_model_branch_and_its_alias_are_the_same_flag():
    p = _parser(family="dual")
    assert (spec_from_args(p.parse_args(["--model-branch", "cnn"])).branch
            == spec_from_args(p.parse_args(["--branch-1d", "cnn"])).branch == "cnn")


def test_underscored_branch_spelling_is_accepted():
    """`--arch cnn_lstm` was the groundmotion flag; it must keep working."""
    args = _parser(family="window").parse_args(["--model", "groundmotion",
                                                "--arch", "cnn_lstm"])
    assert spec_from_args(args).branch == "cnn-lstm"


def test_unknown_branch_names_the_legal_ones():
    args = _parser(family="dual").parse_args(["--model-branch", "transformer"])
    with pytest.raises(ValueError, match="lstm, cnn, cnn-lstm"):
        spec_from_args(args)


def test_branchless_model_rejects_a_branch():
    args = _parser(family="sequence").parse_args(["--model", "gru",
                                                  "--model-branch", "plain"])
    with pytest.raises(ValueError, match="has no branches"):
        spec_from_args(args)


def test_spec_records_only_the_chosen_models_params():
    """`--levels` belongs to tcn; a gru spec must not claim it."""
    p = _parser(family="sequence")
    assert "levels" not in spec_from_args(p.parse_args(["--model", "gru"])).params
    assert "levels" in spec_from_args(p.parse_args(["--model", "tcn"])).params


def test_bad_default_override_is_refused():
    """A renamed flag would otherwise take no effect and say nothing."""
    with pytest.raises(ValueError, match="hiddne"):
        _parser(family="dual", defaults={"hiddne": 48})


def test_family_restricts_what_can_be_chosen():
    """A dual-tensor task must not be able to select a TCN."""
    with pytest.raises(SystemExit):
        _parser(family="dual").parse_args(["--model", "tcn"])
    assert {a.key for a in by_family("sequence")} == {"sequence-head", "gru", "tcn"}
    assert all(a.family in FAMILIES for a in ARCHITECTURES)


def test_spec_round_trips_through_disk(tmp_path):
    spec = spec_from_args(_parser(family="dual").parse_args(
        ["--branch-1d", "cnn-lstm", "--hidden", "48"]))
    spec.save(tmp_path)
    back = ModelSpec.load(tmp_path)
    assert back == spec
    assert json.loads((tmp_path / "model.json").read_text())["branch"] == "cnn-lstm"
    assert ModelSpec.load(tmp_path / "nothing-here") is None


def test_disagreements_catch_a_retyped_spec():
    """The exact failure the 'must match the checkpoints' flags cannot prevent."""
    p = _parser(family="dual")
    trained = spec_from_args(p.parse_args(["--branch-1d", "cnn-lstm", "--hidden", "48"]))
    at_eval = spec_from_args(p.parse_args(["--branch-1d", "lstm", "--hidden", "48"]))
    assert disagreements(trained, at_eval) == {"branch": ("cnn-lstm", "lstm")}
    assert disagreements(trained, trained) == {}


def test_describe_shows_only_what_was_changed():
    spec = spec_from_args(_parser(family="dual").parse_args(["--hidden", "48"]))
    assert spec.describe() == "dual-channel branch=lstm hidden=48"


# ---------------------------------------------------------------------------
# The two converted trainers. These assert that converting them changed nothing
# a recorded command or an existing checkpoint depends on.
# ---------------------------------------------------------------------------

def test_detector_still_accepts_its_historical_flags_and_tag():
    """Every `--channels/--fusion/--branch-1d ...` in experiments/reproduce runs."""
    import os

    from sismokaos.detection.cnn_lstm_classify import parse_args

    argv = ["--dataset-dir", "ds", "--channels", "all", "--fusion", "linear",
            "--branch-1d", "cnn-lstm", "--seq-transform", "asinh",
            "--batch-size", "32"]
    old_argv = sys.argv
    try:
        sys.argv = ["cnn_lstm_classify.py"] + argv
        args = parse_args()
    finally:
        sys.argv = old_argv
    spec = spec_from_args(args)
    # `sismokaos.checkpoints` matches on this string, and quarantined
    # checkpoints are the cost of it changing.
    tag = f"{args.channels}_{args.fusion}_{spec.branch}_{args.seq_transform}"
    assert tag == "all_linear_cnn-lstm_asinh"
    assert (args.hidden, args.fusion_dim, args.dropout) == (48, 96, 0.4)


def test_detector_withholds_aux_channels_it_cannot_honour():
    """--channels 1d+aux would build the 1d network under a different name."""
    from sismokaos.detection.cnn_lstm_classify import parse_args

    old_argv = sys.argv
    try:
        sys.argv = ["cnn_lstm_classify.py", "--dataset-dir", "ds",
                    "--channels", "1d+aux"]
        with pytest.raises(SystemExit):
            parse_args()
    finally:
        sys.argv = old_argv


def test_magnitude_keeps_its_checkpoint_names_and_gains_the_new_flags():
    """Default runs name files exactly as before; new arms get a segment."""
    from sismokaos.magnitude.cnn_lstm_regression import parse_args

    def spec_for(extra):
        old_argv = sys.argv
        try:
            sys.argv = ["cnn_lstm_regression.py", "--dataset-dir", "ds"] + extra
            return spec_from_args(parse_args())
        finally:
            sys.argv = old_argv

    def segment(spec):
        return "".join(f"_{v}" for v in (spec.branch, spec.params["fusion"])
                       if v not in ("lstm", "linear"))

    assert segment(spec_for([])) == "", "existing checkpoint names would change"
    assert segment(spec_for(["--model-branch", "cnn-lstm", "--fusion", "gate"])) \
        == "_cnn-lstm_gate"


@pytest.mark.parametrize("channels,expect", [
    ("all", (True, True, True)),
    ("1d+2d", (True, True, False)),
    ("1d", (True, False, False)),
    ("2d", (False, True, False)),
    ("2d+aux", (False, True, True)),
])
def test_channels_select_the_branches_they_name(channels, expect):
    """`1d+2d` is the deployable one: both waveforms, no catalogue-derived aux.

    `aux = (log_snr, log_distance)` and log_distance is the distance to a
    CATALOGUED hypocentre, which a window the detector just flagged does not
    have. Every other multi-branch value pulls that vector in, so without
    `1d+2d` an operational stage 2 had to drop a waveform branch to drop the
    aux -- confounding the two ablations.
    """
    m = ModelSpec(model="dual-channel", branch="lstm",
                  params={"channels": channels}).build(
        seq_dim=3, img_channels=3, aux_dim=2)
    assert (m.use_1d, m.use_2d, m.use_aux) == expect
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 600, 3), torch.randn(2, 3, 32, 32), torch.randn(2, 2))
    assert tuple(out.shape) == (2, 1)


def test_1d_2d_ignores_the_aux_vector_entirely():
    """Not merely zeroed: the aux path must not exist, so it cannot leak."""
    m = ModelSpec(model="dual-channel", branch="lstm",
                  params={"channels": "1d+2d"}).build(
        seq_dim=3, img_channels=3, aux_dim=2)
    m.eval()
    seq, img = torch.randn(2, 600, 3), torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        a = m(seq, img, torch.zeros(2, 2))
        b = m(seq, img, torch.full((2, 2), 99.0))
    assert torch.equal(a, b), "aux changed the output of a --channels 1d+2d model"


def test_spec_carries_the_split_protocol_beside_the_weights(tmp_path):
    """Geometry is not enough to reproduce a checkpoint's number.

    magnitude_error_profile defaulted to `--split-by both` while the trainer
    defaulted to `event`, so scoring a checkpoint without repeating the flag
    re-derived a DIFFERENT test set than the training run had reported on --
    same weights, same model, a number answering a question nobody asked. The
    protocol now travels in the same file.
    """
    spec = ModelSpec(model="dual-channel", branch="lstm", params={"channels": "all"})
    spec.save(tmp_path, protocol={"split_by": "both", "seed_split": 7,
                                  "detector_manifest": None})

    assert ModelSpec.load(tmp_path) == spec, "extra fields must not disturb the spec"
    proto = ModelSpec.load_extra(tmp_path, "protocol")
    assert proto["split_by"] == "both" and proto["seed_split"] == 7
    assert ModelSpec.load_extra(tmp_path, "nope") is None
    assert ModelSpec.load_extra(tmp_path / "absent", "protocol") is None
