from __future__ import annotations

import argparse
from time import monotonic

from signova.config import DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, ensure_project_dirs
from signova.core.landmarks import LandmarkDetector, cv2
from signova.training.dataset import DatasetManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SIGNOVA landmark samples.")
    parser.add_argument("--label", required=True, help="Sign label, for example HELLO.")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to save.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--interval",
        type=int,
        default=4,
        help="Save one sample every N tracked frames.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Press C to capture instead of saving automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    if cv2 is None:
        raise RuntimeError("OpenCV is not available. Install requirements.txt first.")

    manager = DatasetManager()
    detector = LandmarkDetector()
    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    saved = 0
    tracked_frames = 0
    started_at = monotonic()

    try:
        while saved < args.samples:
            ok, frame = capture.read()
            if not ok:
                break

            detection = detector.process(frame)
            if detection.has_hand:
                tracked_frames += 1
                should_save = (
                    not args.manual
                    and tracked_frames % max(args.interval, 1) == 0
                )
                if should_save:
                    manager.save_sample(
                        args.label,
                        detection.features,
                        {
                            "source": "cli_auto",
                            "tracking_confidence": detection.tracking_confidence,
                        },
                    )
                    saved += 1

            status = (
                f"LABEL {args.label.upper()} | SAVED {saved}/{args.samples} | "
                f"TRACKING {int(detection.tracking_confidence * 100)}%"
            )
            cv2.rectangle(detection.frame, (16, 16), (860, 58), (5, 12, 24), -1)
            cv2.putText(
                detection.frame,
                status,
                (28, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 229, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("SIGNOVA Dataset Collector", detection.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if args.manual and key == ord("c") and detection.has_hand:
                manager.save_sample(
                    args.label,
                    detection.features,
                    {
                        "source": "cli_manual",
                        "tracking_confidence": detection.tracking_confidence,
                    },
                )
                saved += 1
    finally:
        detector.close()
        capture.release()
        cv2.destroyAllWindows()

    elapsed = max(monotonic() - started_at, 0.1)
    print(f"Saved {saved} samples for {args.label.upper()} in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
