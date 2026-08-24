"""
The weighting algorithm: combines the lip-reading model's prediction with
the throat-vibration confidence signal into one final output for the demo.

Design, and why:
- The lip model is the PRIMARY signal -- it's the one actually classifying
  words from an 8-way softmax. Vibration is a SECONDARY signal that (a)
  nudges the displayed confidence number, and (b) can break ties when the
  lip model's top-2 guesses are both in a known "these look the same on
  the mouth" set for your vocabulary.
- Your current 8-word vocab (hello/yes/no/thanks/stop/help/good/bye) wasn't
  chosen around p/b/m-style visual confusability the way the original
  brief's "pain/brain/main" example was. CONFUSABLE_PAIRS is empty-ish by
  default -- fill it in once you've actually seen which words the trained
  model confuses (check its confusion matrix from train.py's evaluation,
  or just watch which words get swapped during testing).
"""

from typing import List

# Add pairs here once you know which words actually get confused in
# practice, e.g. frozenset({"stop", "no"}) if the model keeps mixing them
# up. Vibration confidence only gets used to break a tie for pairs listed
# here -- otherwise the lip model's top-1 guess is trusted as-is.
CONFUSABLE_PAIRS = set()

# If the lip model's top-1 and top-2 confidence are within this margin of
# each other AND the pair is in CONFUSABLE_PAIRS, let vibration decide.
AMBIGUITY_MARGIN = 0.15


def resolve_ambiguous(lip_candidates: List[str], vibration_confidence: float) -> str:
    """
    lip_candidates: 2+ words the lip model is torn between.
    vibration_confidence: 0-1 score from ThroatVibrationSensor.read().

    Placeholder bucket rule -- replace with something trained on real
    (lip_prediction, vibration_signal) -> word pairs once you have labeled
    examples of your actual confusable words.
    """
    if not lip_candidates:
        return ""
    if len(lip_candidates) == 1:
        return lip_candidates[0]

    if vibration_confidence < 0.33:
        bucket = 0
    elif vibration_confidence < 0.66:
        bucket = 1
    else:
        bucket = 2

    return lip_candidates[min(bucket, len(lip_candidates) - 1)]


def combined_confidence(lip_confidence: float, vibration_confidence: float,
                         lip_weight: float = 0.8) -> float:
    """
    Weighted blend for the displayed confidence number. lip_weight=0.8
    because the lip model is doing the actual classification -- vibration
    is a supporting signal, not an independent vote. Tune this weight once
    you've watched both signals side by side during a live test; if
    vibration confidence is noisy/unreliable relative to the lip model,
    push lip_weight higher (e.g. 0.9).
    """
    vib_weight = 1.0 - lip_weight
    return (lip_confidence * lip_weight) + (vibration_confidence * vib_weight)


def fuse_prediction(lip_result: dict, vibration_confidence: float) -> dict:
    """
    Main entry point called once per /signals request.

    lip_result: the dict returned by LipReader.process() -- has
        word, confidence, top2_word, top2_confidence, face_points
    vibration_confidence: 0-1 from ThroatVibrationSensor.read()['confidence']

    Returns: {"word": str, "confidence": float}
    """
    word = lip_result["word"]
    lip_conf = lip_result["confidence"]

    pair = frozenset({lip_result["word"], lip_result["top2_word"]})
    margin = lip_conf - lip_result["top2_confidence"]

    if pair in CONFUSABLE_PAIRS and margin < AMBIGUITY_MARGIN:
        word = resolve_ambiguous(
            [lip_result["word"], lip_result["top2_word"]],
            vibration_confidence,
        )

    final_confidence = combined_confidence(lip_conf, vibration_confidence)
    return {"word": word, "confidence": final_confidence}
