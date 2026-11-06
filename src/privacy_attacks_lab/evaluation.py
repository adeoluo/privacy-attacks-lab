"""Leave-one-out evaluation of membership inference attacks on a family of models.

Each model in turn plays the target; the remaining models are its reference
(shadow) models. For the target, the "members" are its own training examples and
the "non-members" the rest of the pool, so the prior is 50% as in the LiRA paper.
Results are averaged over targets; the spread across targets is reported next to
a bootstrap interval of the first target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from privacy_attacks_lab.attacks import (
    ShadowAttack,
    calibrated_attack,
    lira_offline,
    lira_online,
    loss_attack,
    shadow_features,
)
from privacy_attacks_lab.metrics import DEFAULT_FPRS, RocSummary, summarize
from privacy_attacks_lab.shadow import Signals

#: Attack variants evaluated by default (name -> description).
VARIANTS: dict[str, str] = {
    "loss": "loss threshold (Yeom et al. 2018)",
    "calibrated": "difficulty calibrated loss, OUT references (Watson et al. 2022)",
    "shadow": "per-class shadow attack models (Shokri et al. 2017)",
    "lira-offline": "LiRA offline, one-sided, per-example variance",
    "lira-offline-fixed": "LiRA offline, one-sided, pooled variance",
    "lira-online": "LiRA online, per-example variance",
    "lira-online-fixed": "LiRA online, pooled variance",
}

#: The authors' released LiRA scoring (:mod:`privacy_attacks_lab.attacks.lira_reference`).
REFERENCE_VARIANTS: dict[str, str] = {
    "lira-online-ref": "LiRA online, authors' released scoring",
    "lira-online-ref-fixed": "LiRA online, authors' released scoring, global variance",
    "lira-offline-ref": "LiRA offline, authors' released scoring (two-sided)",
    "lira-offline-ref-fixed": "LiRA offline, authors' released scoring, global variance",
}


def attack_scores(
    sig: Signals,
    target: int,
    variant: str,
    refs: Sequence[int] | None = None,
    shadow_models: int = 8,
) -> np.ndarray:
    """Scores of ``variant`` against model ``target`` using reference models ``refs``."""
    refs_arr = (
        np.array([i for i in range(sig.num_models) if i != target])
        if refs is None
        else np.asarray(refs)
    )
    if target in set(refs_arr.tolist()):
        raise ValueError("the target cannot be its own reference model")
    t_loss, t_phi = sig.loss[target], sig.phi[target]
    r_phi, r_masks = sig.phi[refs_arr], sig.masks[refs_arr]
    if variant == "loss":
        return loss_attack(t_loss)
    if variant == "calibrated":
        return calibrated_attack(t_loss, sig.loss[refs_arr], r_masks)
    if variant == "calibrated-online":
        return calibrated_attack(t_loss, sig.loss[refs_arr], r_masks, online=True)
    if variant == "shadow":
        use = refs_arr[:shadow_models]
        attack = ShadowAttack().fit(
            shadow_features(sig.topk[use], sig.phi[use]), sig.masks[use], sig.labels
        )
        return attack.score(shadow_features(sig.topk[target], t_phi), sig.labels)
    if variant.endswith("-ref") or variant.endswith("-ref-fixed"):
        from privacy_attacks_lab.attacks import lira_reference as ref

        fixed = variant.endswith("-fixed")
        t3 = t_phi[:, None] if t_phi.ndim == 1 else t_phi
        r3 = r_phi[:, :, None] if r_phi.ndim == 2 else r_phi
        if variant.startswith("lira-online"):
            return ref.reference_online(r_masks, r3, t3, fix_variance=fixed)
        if variant.startswith("lira-offline"):
            return ref.reference_offline(r_masks, r3, t3, fix_variance=fixed)
    if variant.startswith("lira-"):
        fixed = variant.endswith("-fixed")
        if variant.startswith("lira-online"):
            return lira_online(t_phi, r_phi, r_masks, fix_variance=fixed)
        if variant.startswith("lira-offline-pdf"):
            return lira_offline(t_phi, r_phi, r_masks, fix_variance=fixed, one_sided=False)
        if variant.startswith("lira-offline"):
            return lira_offline(t_phi, r_phi, r_masks, fix_variance=fixed)
    raise ValueError(f"unknown attack variant {variant!r}")


@dataclass
class AttackResult:
    """Per-target summaries of one attack variant."""

    variant: str
    summaries: list[RocSummary] = field(default_factory=list)

    def aggregate(self) -> dict[str, Any]:
        table = np.array([[s.auc, s.accuracy, *s.tprs] for s in self.summaries])
        keys = ["auc", "accuracy", *[f"tpr@{f:g}" for f in self.summaries[0].fprs]]
        out: dict[str, Any] = {"variant": self.variant, "targets": len(self.summaries)}
        for j, k in enumerate(keys):
            out[k] = float(table[:, j].mean())
            out[f"{k}_std"] = float(table[:, j].std(ddof=1)) if len(table) > 1 else 0.0
        return out


def evaluate(
    sig: Signals,
    targets: Sequence[int],
    variants: Sequence[str] = tuple(VARIANTS),
    num_refs: int | None = None,
    fprs: Sequence[float] = DEFAULT_FPRS,
    shadow_models: int = 8,
) -> dict[str, AttackResult]:
    """Evaluate every variant against every target with ``num_refs`` references each."""
    results = {v: AttackResult(v) for v in variants}
    for t in targets:
        others = [i for i in range(sig.num_models) if i != t]
        refs = others if num_refs is None else others[:num_refs]
        labels = sig.masks[t]
        for v in variants:
            scores = attack_scores(sig, t, v, refs, shadow_models=shadow_models)
            results[v].summaries.append(summarize(scores, labels, fprs))
    return results
