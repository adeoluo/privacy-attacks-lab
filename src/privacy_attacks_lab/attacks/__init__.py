"""Membership inference attacks.

Every attack maps the target model's per-example signals (and, where it needs them,
the signals of reference or shadow models with known membership) to one score per
example; higher means "more likely a member". Scores feed
:mod:`privacy_attacks_lab.metrics` unchanged.

=================  ===========================================  ==========================
attack             idea                                         reference
=================  ===========================================  ==========================
``loss``           members have lower loss                      Yeom et al. 2018
``calibrated``     loss minus the example's mean OUT loss       Watson et al. 2022
``shadow``         classifier on shadow models' outputs         Shokri et al. 2017
``lira-offline``   one-sided test against the OUT Gaussian      Carlini et al. 2022
``lira-online``    likelihood ratio of IN and OUT Gaussians     Carlini et al. 2022
=================  ===========================================  ==========================
"""

from __future__ import annotations

from privacy_attacks_lab.attacks.calibrated import calibrated_attack
from privacy_attacks_lab.attacks.lira import GaussianFit, fit_gaussians, lira_offline, lira_online
from privacy_attacks_lab.attacks.loss import loss_attack
from privacy_attacks_lab.attacks.shadow import ShadowAttack, shadow_features

ATTACKS: tuple[str, ...] = (
    "loss",
    "calibrated",
    "shadow",
    "lira-offline",
    "lira-online",
)

__all__ = [
    "ATTACKS",
    "GaussianFit",
    "ShadowAttack",
    "calibrated_attack",
    "fit_gaussians",
    "lira_offline",
    "lira_online",
    "loss_attack",
    "shadow_features",
]
