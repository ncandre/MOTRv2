"""Launch a detection algorithm on a video."""
import argparse
import sys
from pathlib import Path

from loguru import logger

from saruman.common.constant import DETECT_AND_TRACK_CONFIG_NAME
from saruman.common.file_utils import read_yaml, save_yaml
from saruman.common.metadata_reader import delete_temporary_folder
from saruman.common.timer import Timer
from saruman.predict.detect_and_track import detect_and_track
from saruman.predict.detector import Detector, ModelDetector, OracleDetector
from saruman.predict.feature_extractor import FeatureExtractor
from saruman.schemas.detect_and_track import CACHE_SUCCESS_FN, DetectAndTrackConfig
from saruman.tracking.tracker import Tracker
from saruman.utils.monitor import MonitorPipe
from saruman.utils.save import SaveOrchestrator

logger.remove()
logger.add(sys.stderr, level="INFO")

detector: Detector | None = None
feature_extractor: FeatureExtractor | None = None


def track_video(
    detect_and_track_config: DetectAndTrackConfig, video_path: Path, backup_output_dir: Path | None = None
) -> None:
    """Apply tracking algorithm to a single video.

    Args:
        detect_and_track_config: config holding the params to use for the deepsort.
        video_path: path to the video to apply the algo on.
        backup_output_dir: dir where a backup of the detect_and_track_config is saved as a .yaml.
            If None backup is not made.
    """
    global detector
    global feature_extractor
    if backup_output_dir is not None:
        save_yaml(detect_and_track_config.dict(), backup_output_dir / DETECT_AND_TRACK_CONFIG_NAME)
    else:
        logger.warning("The config given will not be backed up as no output_dir was given.")

    with Timer() as global_timer:
        if detector is None and detect_and_track_config.detection is not None:
            # a normal detector is used
            detector = ModelDetector(detect_and_track_config.detection)

        if detect_and_track_config.feature_extractor is not None and feature_extractor is None:
            feature_extractor = FeatureExtractor(detect_and_track_config.feature_extractor)

        tracker = Tracker(detect_and_track_config.tracker)
        monitor = MonitorPipe(detect_and_track_config.monitor.enabled)
        save_orchestrator = SaveOrchestrator(
            detect_and_track_config.outputs, video_path, use_multiprocessing=detect_and_track_config.use_multiprocessing
        )
        if isinstance(detector, ModelDetector) and detector.model is None:
            logger.info("Loading detector model")
            with Timer() as t:
                detector.load_model()
            monitor.update_detector_loading(t.elapsed_time)
        if feature_extractor is not None and feature_extractor.shortened_model is None:
            logger.info("Loading feature extractor model")
            with Timer() as t:
                feature_extractor.load_model()
            monitor.update_feature_extractor_loading(t.elapsed_time)
        for _ in detect_and_track(
            video_path,
            detector,
            save_orchestrator,
            monitor,
            tracker,
            feature_extractor=feature_extractor,
            use_multiprocessing=detect_and_track_config.use_multiprocessing,
            nb_frames_skipped=detect_and_track_config.nb_frames_skipped,
        ):
            pass
        save_orchestrator.close()

    delete_temporary_folder()
    monitor.save_recaps(video_path, detect_and_track_config, global_timer.elapsed_time)


def main(
    detect_and_track_config: DetectAndTrackConfig, video_folder: Path, backup_output_dir: Path | None = None
) -> None:
    """Launch tracking on the videos of the given folder.

    Args:
        detect_and_track_config: config holding the params to use for the deepsort.
        video_folder: path to the video folder to apply the algo on.
        backup_output_dir: dir where a backup of the detect_and_track_config is saved as a .yaml.
            If None backup is not made.
    """
    global detector
    # a detection file is used to do prediction
    if detect_and_track_config.oracle_detection is not None:
        detector = OracleDetector(detect_and_track_config.oracle_detection)
        detector.check_gt_file_for_each_video(video_folder)
    for video_path in video_folder.glob("*"):
        logger.info(f"Processing {video_path} ...")
        track_video(detect_and_track_config, video_path, backup_output_dir)

    if (
        detect_and_track_config.detection is not None
        and detect_and_track_config.detection.path_for_detector_cache is not None
    ):
        (detect_and_track_config.detection.path_for_detector_cache / CACHE_SUCCESS_FN).touch()
    MonitorPipe.aggregate_global_reports(detect_and_track_config.monitor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FMV prediction script.")
    parser.add_argument(
        "--detect_config",
        "-d",
        type=Path,
        help="yaml conf for the detection.",
        required=True,
    )
    parser.add_argument(
        "--video_folder",
        "-v",
        type=Path,
        help="path to the video folder to apply the algo on.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        help="folder where all outputs will be saved.",
        required=False,
        default=None,
    )
    args = parser.parse_args()

    detect_and_track_conf = DetectAndTrackConfig.overwrite_save_output_dir(
        read_yaml(args.detect_and_track_config), args.output_dir
    )
    main(DetectAndTrackConfig(**detect_and_track_conf), args.video_folder, backup_output_dir=args.output_dir)
