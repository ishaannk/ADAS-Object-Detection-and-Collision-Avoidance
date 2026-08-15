import pytest

from adas.risk.ttc import BRAKE_TTC_S, WARN_TTC_S, TrackHistory, assess_risk


def test_closing_speed_from_approaching_object():
    history = TrackHistory()
    # Closing at a steady 10 m/s.
    for t, d in [(0.0, 50.0), (0.1, 49.0), (0.2, 48.0), (0.3, 47.0)]:
        history.update(t, d)

    speed = history.closing_speed_mps()
    assert speed == pytest.approx(10.0, abs=0.05)


def test_closing_speed_from_receding_object_is_negative():
    history = TrackHistory()
    for t, d in [(0.0, 10.0), (0.1, 11.0), (0.2, 12.0)]:
        history.update(t, d)

    assert history.closing_speed_mps() < 0


def test_closing_speed_needs_at_least_two_observations():
    history = TrackHistory()
    history.update(0.0, 10.0)
    assert history.closing_speed_mps() is None


def test_assess_risk_tiers():
    # 3m at 10 m/s closing -> TTC = 0.3s -> brake
    brake = assess_risk(track_id=1, distance_m=3.0, closing_speed_mps=10.0)
    assert brake.tier == "brake"
    assert brake.ttc_s < BRAKE_TTC_S

    # 20m at 10 m/s closing -> TTC = 2.0s -> warn
    warn = assess_risk(track_id=2, distance_m=20.0, closing_speed_mps=10.0)
    assert warn.tier == "warn"
    assert BRAKE_TTC_S < warn.ttc_s <= WARN_TTC_S

    # Far away and slow closing -> monitor
    monitor = assess_risk(track_id=3, distance_m=100.0, closing_speed_mps=1.0)
    assert monitor.tier == "monitor"

    # Receding object (no closing speed) -> monitor, no TTC
    receding = assess_risk(track_id=4, distance_m=5.0, closing_speed_mps=-3.0)
    assert receding.tier == "monitor"
    assert receding.ttc_s is None
