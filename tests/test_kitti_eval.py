from adas.eval.kitti_eval import GroundTruthBox, PredBox, average_precision, iou, matches_difficulty


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_difficulty_tiers_by_height_occlusion_truncation():
    easy = GroundTruthBox(cls="Car", bbox=(0, 0, 50, 50), occluded=0, truncated=0.0)
    assert matches_difficulty(easy, "easy")

    moderate_only = GroundTruthBox(cls="Car", bbox=(0, 0, 50, 30), occluded=1, truncated=0.2)
    assert not matches_difficulty(moderate_only, "easy")
    assert matches_difficulty(moderate_only, "moderate")

    too_small_for_any_tier = GroundTruthBox(cls="Car", bbox=(0, 0, 50, 10), occluded=0, truncated=0.0)
    assert not matches_difficulty(too_small_for_any_tier, "hard")


def test_average_precision_perfect_detector_is_one():
    gts_by_frame = [[GroundTruthBox(cls="Car", bbox=(0, 0, 50, 50), occluded=0, truncated=0.0)]]
    preds = [PredBox(cls="Car", bbox=(0, 0, 50, 50), confidence=0.9, frame=0)]

    ap = average_precision(preds, gts_by_frame, cls="Car", difficulty="easy")
    assert ap == 1.0


def test_average_precision_no_predictions_is_zero():
    gts_by_frame = [[GroundTruthBox(cls="Car", bbox=(0, 0, 50, 50), occluded=0, truncated=0.0)]]
    ap = average_precision([], gts_by_frame, cls="Car", difficulty="easy")
    assert ap == 0.0


def test_average_precision_wrong_class_prediction_scores_zero():
    gts_by_frame = [[GroundTruthBox(cls="Car", bbox=(0, 0, 50, 50), occluded=0, truncated=0.0)]]
    preds = [PredBox(cls="Pedestrian", bbox=(0, 0, 50, 50), confidence=0.9, frame=0)]
    ap = average_precision(preds, gts_by_frame, cls="Car", difficulty="easy")
    assert ap == 0.0


def test_correct_detection_of_a_harder_instance_does_not_penalize_easy_ap():
    """Regression test: a real, heavily-occluded Car (moderate/hard-only) that
    the detector correctly finds must not count as a false positive when
    scoring the "easy" tier, which has no matching GT box for it."""
    gts_by_frame = [
        [
            GroundTruthBox(cls="Car", bbox=(0, 0, 50, 50), occluded=0, truncated=0.0),  # easy
            GroundTruthBox(cls="Car", bbox=(100, 100, 130, 130), occluded=2, truncated=0.4),  # hard-only
        ]
    ]
    preds = [
        PredBox(cls="Car", bbox=(0, 0, 50, 50), confidence=0.9, frame=0),
        PredBox(cls="Car", bbox=(100, 100, 130, 130), confidence=0.8, frame=0),
    ]

    ap_easy = average_precision(preds, gts_by_frame, cls="Car", difficulty="easy")
    assert ap_easy == 1.0  # the hard-only detection is ignored, not penalized


def test_difficulty_ordering_is_not_inverted_when_detector_performs_uniformly():
    """A detector that finds every GT box regardless of difficulty should not
    score worse on 'easy' than on 'hard' — that would indicate the classic
    ignore-region bug (real detections of harder instances counted as FPs
    against the easy tier)."""
    gts_by_frame = [
        [
            GroundTruthBox(cls="Car", bbox=(0, 0, 50, 50), occluded=0, truncated=0.0),  # easy
            GroundTruthBox(cls="Car", bbox=(100, 0, 130, 30), occluded=1, truncated=0.2),  # moderate-only
            GroundTruthBox(cls="Car", bbox=(200, 0, 230, 30), occluded=2, truncated=0.4),  # hard-only
        ]
    ]
    preds = [
        PredBox(cls="Car", bbox=(0, 0, 50, 50), confidence=0.9, frame=0),
        PredBox(cls="Car", bbox=(100, 0, 130, 30), confidence=0.85, frame=0),
        PredBox(cls="Car", bbox=(200, 0, 230, 30), confidence=0.8, frame=0),
    ]

    ap_easy = average_precision(preds, gts_by_frame, cls="Car", difficulty="easy")
    ap_moderate = average_precision(preds, gts_by_frame, cls="Car", difficulty="moderate")
    ap_hard = average_precision(preds, gts_by_frame, cls="Car", difficulty="hard")

    assert ap_easy == ap_moderate == ap_hard == 1.0
