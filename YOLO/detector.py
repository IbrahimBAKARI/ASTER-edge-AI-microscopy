"""YOLO WBC detection boundary.

Keeping this adapter in its own package makes the detection stage replaceable
without coupling the desktop interface to Ultralytics.
"""

from aster_pipeline.yolo_detector import YoloWBCDetector


class WBCDetector(YoloWBCDetector):
    """Stable public name for the deployed single-class WBC detector."""

