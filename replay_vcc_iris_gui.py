"""独立 GUI 重播：从已有 experiment JSON 直接播放轨迹，无需重新规划。

用法:
    python replay_vcc_iris_gui.py                           # 使用默认路径
    python replay_vcc_iris_gui.py path/to/experiment.json   # 指定文件
    python replay_vcc_iris_gui.py --speed 0.1               # 调整每帧间隔(秒)
    python replay_vcc_iris_gui.py --hold 10                 # 播放结束后保持窗口(秒)
    python replay_vcc_iris_gui.py --video out.mp4           # 回放并录制视频
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EXPERIMENT_JSON = str(
    REPO_ROOT / "artifacts" / "sdf_exp" / "vcc_iris_experiment.json"
)


def main():
    parser = argparse.ArgumentParser(description="VCC+IRIS-ZO 轨迹 GUI 重播")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=DEFAULT_EXPERIMENT_JSON,
        help="experiment JSON 文件路径 (默认: artifacts/sdf_exp/vcc_iris_experiment.json)",
    )
    parser.add_argument("--speed", type=float, default=0.15, help="每帧间隔秒数 (默认 0.15)")
    parser.add_argument("--hold", type=float, default=5.0, help="播放结束后保持窗口秒数 (默认 5.0)")
    parser.add_argument("--video", type=str, default=None, help="可选：输出 mp4 路径")
    parser.add_argument("--fps", type=int, default=15, help="录制视频帧率 (默认 15)")
    args = parser.parse_args()

    from CBF_experiment.active.pybullet.self_collision.vcc_iris.io.gui import replay_from_json

    replay_from_json(
        args.json_path,
        sleep_dt=args.speed,
        hold_seconds=args.hold,
        video_output_path=args.video,
        video_fps=args.fps,
    )


if __name__ == "__main__":
    main()
