"""Time-to-collision risk assessment.

The original prototype (docs/PLAN.md, DTC-04) flagged risk from a single
frame's static distance threshold, so a stationary object 8m away and one
closing fast from 8m looked identical. This module instead tracks each
object's distance across frames and computes time-to-collision (TTC) from
its closing speed, only alerting on objects that are actually approaching.

Alert tiers are shaped after published AEB test-protocol scenarios (Euro
NCAP AEB Car-to-Car / Vulnerable Road User): a short TTC band where braking
alone can no longer avoid impact ("brake"), a longer band where there is
still time for a warning or an evasive steer ("warn"), and everything else
monitored but not alerted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

BRAKE_TTC_S = 1.5
WARN_TTC_S = 3.0
MIN_CLOSING_SPEED_MPS = 0.3  # ignore noise-level closing speed, avoid false positives


@dataclass
class TrackHistory:
    """Rolling (timestamp, distance) observations for one tracked object."""

    max_len: int = 5
    observations: deque = field(default_factory=lambda: deque(maxlen=5))

    def update(self, timestamp_s: float, distance_m: float) -> None:
        self.observations.append((timestamp_s, distance_m))

    def closing_speed_mps(self) -> float | None:
        """Positive = approaching. Estimated by least-squares slope of distance vs. time."""
        if len(self.observations) < 2:
            return None
        ts = [t for t, _ in self.observations]
        ds = [d for _, d in self.observations]
        t0 = ts[0]
        ts = [t - t0 for t in ts]
        n = len(ts)
        mean_t = sum(ts) / n
        mean_d = sum(ds) / n
        num = sum((t - mean_t) * (d - mean_d) for t, d in zip(ts, ds))
        den = sum((t - mean_t) ** 2 for t in ts)
        if den == 0:
            return None
        slope = num / den  # d(distance)/dt: negative while approaching
        return -slope


@dataclass
class RiskAssessment:
    track_id: int
    distance_m: float
    closing_speed_mps: float | None
    ttc_s: float | None
    tier: str  # "brake" | "warn" | "monitor"
    guidance: str


def assess_risk(track_id: int, distance_m: float, closing_speed_mps: float | None) -> RiskAssessment:
    if closing_speed_mps is None or closing_speed_mps < MIN_CLOSING_SPEED_MPS:
        return RiskAssessment(track_id, distance_m, closing_speed_mps, None, "monitor", "No closing risk")

    ttc_s = distance_m / closing_speed_mps

    if ttc_s <= BRAKE_TTC_S:
        return RiskAssessment(track_id, distance_m, closing_speed_mps, ttc_s, "brake", "Trigger Brake")
    if ttc_s <= WARN_TTC_S:
        return RiskAssessment(track_id, distance_m, closing_speed_mps, ttc_s, "warn", "Forward Collision Warning")
    return RiskAssessment(track_id, distance_m, closing_speed_mps, ttc_s, "monitor", "Monitor")
