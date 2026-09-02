"""
tests/test_ablation_settings_patch.py
Regression guard for the bug documented in GAPS_AND_IMPROVEMENTS.md item A.1 /
the "Fix ablation study never actually applying its config overrides" commit.

Every ablation-relevant flag is imported into its consuming module with
`from config.settings import X` (a bound copy, not a live reference), so patching
config.settings alone does nothing. _SettingsPatch must patch each dependent
module's own copy directly. If this regresses, every ablation config silently
runs with identical real pipeline behavior — exactly what happened before this
fix was found, mid-run, on 2026-08-20.
"""

from evaluation.ablation import _SettingsPatch, _DEPENDENT_MODULES
import pipeline.nodes.validator as validator_mod
import pipeline.nodes.critic as critic_mod
import pipeline.router as router_mod


def test_patch_propagates_to_dependent_module_bound_copies():
    assert validator_mod.USE_VALIDATOR is True  # sanity: default is on

    with _SettingsPatch({"USE_VALIDATOR": False}):
        assert validator_mod.USE_VALIDATOR is False

    assert validator_mod.USE_VALIDATOR is True, "must restore on exit"


def test_patch_affects_all_modules_that_import_the_flag():
    with _SettingsPatch({"USE_CRITIC": False}):
        assert critic_mod.USE_CRITIC is False
        assert router_mod.USE_CRITIC is False

    assert critic_mod.USE_CRITIC is True
    assert router_mod.USE_CRITIC is True


def test_patch_restores_original_value_even_after_exception():
    original = validator_mod.USE_VALIDATOR
    try:
        with _SettingsPatch({"USE_VALIDATOR": False}):
            raise RuntimeError("simulated failure mid-ablation-run")
    except RuntimeError:
        pass
    assert validator_mod.USE_VALIDATOR == original


def test_multiple_overrides_applied_together():
    with _SettingsPatch({"USE_VALIDATOR": False, "USE_CRITIC": False}):
        assert validator_mod.USE_VALIDATOR is False
        assert critic_mod.USE_CRITIC is False
    assert validator_mod.USE_VALIDATOR is True
    assert critic_mod.USE_CRITIC is True


def test_dependent_modules_registry_is_not_empty():
    """A regression where _DEPENDENT_MODULES silently loses an entry would make
    that module's flags un-patchable again without any visible error."""
    assert len(_DEPENDENT_MODULES) >= 6
