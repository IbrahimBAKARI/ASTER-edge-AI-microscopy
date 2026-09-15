# YOLO

This package owns WBC localisation. `WBCDetector` accepts microscopy fields,
writes annotated images and cell crops, and returns detections to the backend.
It does not perform a leukemia diagnosis.

The deployed weight file is `../models/yolo/wbc_detector.pt` and its runtime
settings live in `../config/inference.yaml`.
