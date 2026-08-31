#!/usr/bin/env python3
"""用 Rerun 动态查看 replay_buffer.zarr 中的机械臂末端轨迹。

本脚本把采集得到的末端位姿写成 Rerun 可回放的 3D 场景：
  - 黄色折线：整条末端轨迹
  - 白点 origin：当前末端位置
  - 红/绿/蓝箭头 frame：当前末端坐标系（类似 RViz TF）
  - 青色两指 + 开口线：物理开口（本数据集 zarr 原值越大越夹紧，默认已取反）
  - plots/*/gripper_width_stored：zarr 原值
  - plots/*/gripper_opening：用于显示的物理开口
  - cameras/*：同步相机图像

本机有显示器:
    conda activate Lingbot
    python scripts/visualize_trajectory_rerun.py --episode 0

    python scripts/visualize_trajectory_rerun.py \
  --zarr_path data/cup_demo/replay_buffer.zarr \
  --episode 0
SSH 远程（推荐 Web Viewer）:
    # 终端 1：服务器上启动
    python scripts/visualize_trajectory_rerun.py --episode 0 --web

    # 终端 2：本机做端口转发（两个端口都要转）
    ssh -L 9090:127.0.0.1:9090 -L 9876:127.0.0.1:9876 user@server

    # 本机浏览器打开带连接参数的地址（不要只开 9090）
    # http://127.0.0.1:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy

也可先导出 .rrd，再在本机打开:
    python scripts/visualize_trajectory_rerun.py --episode 0 --save_rrd /tmp/ep0.rrd
    # scp 到本机后: rerun ep0.rrd

Author: Kian
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import rerun as rr
import zarr
from scipy.spatial.transform import Rotation


# 末端坐标轴颜色：X 红 / Y 绿 / Z 蓝（与常见机器人可视化习惯一致）
AXIS_COLORS = np.array(
    [
        [255, 60, 60],   # X: red
        [60, 220, 60],   # Y: green
        [60, 120, 255],  # Z: blue
    ],
    dtype=np.uint8,
)

# 夹爪可视化几何（米）。开口沿局部 X，厚度沿 Y，与 solve_table_collision 一致。
FINGER_THICKNESS = 25.5 / 1000
FINGER_LENGTH = 0.04
FINGER_PAD = 0.006
# 物理全开参考（米）。本数据集 robot0_gripper_width 是夹爪电机行程，
# 越大越夹紧，不是 UMI 那种“开口宽度”；3D 显示默认用 open_ref - stored 还原开口。
GRIPPER_OPEN_REF = 0.08


def episode_slice(episode_ends: np.ndarray, episode: int) -> slice:
    """根据 meta/episode_ends 计算某个 episode 在 data 数组中的切片。

    Zarr replay buffer 会把所有 episode 的时间步沿第 0 维拼接存放，
    episode_ends[i] 表示第 i 条 episode 结束后的全局步数（开区间上界）。

    例如 episode_ends = [873, 1600]：
      episode 0 → slice(0, 873)
      episode 1 → slice(873, 1600)
    """
    if episode < 0 or episode >= len(episode_ends):
        raise ValueError(
            f"Episode {episode} 不存在；有效范围是 0..{len(episode_ends) - 1}"
        )
    start = 0 if episode == 0 else int(episode_ends[episode - 1])
    end = int(episode_ends[episode])
    return slice(start, end)


def find_robot_indices(data) -> list[int]:
    """扫描 data 键名，找出所有 robot{{i}}_eef_pos 对应的机械臂编号。

    例如存在 robot0_eef_pos、robot1_eef_pos 时返回 [0, 1]。
    """
    indices = []
    for key in data.keys():
        if key.startswith("robot") and key.endswith("_eef_pos"):
            # 从 "robot0_eef_pos" 中抽出中间的数字 "0"
            index_text = key[len("robot") : -len("_eef_pos")]
            if index_text.isdigit():
                indices.append(int(index_text))
    return sorted(set(indices))


def find_camera_keys(data) -> list[str]:
    """扫描 data 键名，找出所有 camera*_rgb 图像字段。"""
    return sorted(
        key
        for key in data.keys()
        if key.startswith("camera") and key.endswith("_rgb")
    )


def stored_to_opening(stored_width: float, invert: bool) -> float:
    """把 zarr 里的 gripper_width 转成用于 3D 显示的物理开口。

    collect_drag_umi 走 SingleArm：joint[6]/5*0.11 直接写成 robot0_gripper_width。
    对照腕部图像：stored≈0.03 时两指张开，stored≈0.07 时夹紧，所以默认取反。
    """
    stored_width = float(stored_width)
    if not invert:
        return max(0.0, stored_width)
    return max(0.0, GRIPPER_OPEN_REF - stored_width)


def gripper_openness_color(opening: float) -> list[int]:
    """按物理开口上色：闭合偏橙，张开偏青。"""
    open_ratio = float(np.clip(opening / GRIPPER_OPEN_REF, 0.0, 1.0))
    return [
        int(255 * (1.0 - open_ratio) + 40 * open_ratio),
        int(90 + 130 * open_ratio),
        int(40 + 215 * open_ratio),
    ]


def log_gripper(
    entity_base: str,
    position: np.ndarray,
    rotvec: np.ndarray,
    opening: float,
    stored_width: float,
    axis_length: float,
) -> None:
    """在末端附近画平行夹爪：两块指垫 + 开口连线。

    局部系约定：
      X = 开口方向（两指分别在 ±opening/2）
      Y = 指垫厚度方向
      Z = 指尖伸出方向（从 TCP 往前）
    """
    rotation = Rotation.from_rotvec(rotvec)
    rotation_matrix = rotation.as_matrix()
    quat_xyzw = rotation.as_quat()  # scipy: (x, y, z, w)
    color = gripper_openness_color(opening)

    half_x = FINGER_PAD / 2.0
    half_y = FINGER_THICKNESS / 2.0
    half_z = FINGER_LENGTH / 2.0
    local_centers = np.array(
        [
            [-opening / 2.0, 0.0, half_z],
            [opening / 2.0, 0.0, half_z],
        ]
    )
    world_centers = (rotation_matrix @ local_centers.T).T + position

    rr.log(
        f"{entity_base}/gripper/fingers",
        rr.Boxes3D(
            centers=world_centers,
            half_sizes=[[half_x, half_y, half_z], [half_x, half_y, half_z]],
            quaternions=[quat_xyzw, quat_xyzw],
            colors=[color, color],
            fill_mode=rr.components.FillMode.Solid,
        ),
    )

    # 开口连线画在指尖平面；标签同时标物理开口和 zarr 原值
    local_tips = np.array(
        [
            [-opening / 2.0, 0.0, FINGER_LENGTH],
            [opening / 2.0, 0.0, FINGER_LENGTH],
        ]
    )
    world_tips = (rotation_matrix @ local_tips.T).T + position
    rr.log(
        f"{entity_base}/gripper/opening",
        rr.LineStrips3D(
            [world_tips],
            colors=[color],
            radii=max(axis_length * 0.015, 0.0008),
            labels=[
                f"open {opening * 1000:.1f} mm  (stored {stored_width * 1000:.1f})"
            ],
        ),
    )


def image_as_uint8(image: np.ndarray) -> np.ndarray:
    """把图像转成 Rerun 友好的 uint8 RGB。

    采集时可能存成 float32（范围约 0~1 或 0~255），这里统一规范到
    uint8 的 0~255，避免 Viewer 里发黑/发白。
    """
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        finite_max = float(np.nanmax(image)) if image.size else 0.0
        # 若最大值很小，认为是归一化到 [0,1] 的图像
        if finite_max <= 1.5:
            image = image * 255.0
    return np.nan_to_num(image, nan=0.0).clip(0, 255).astype(np.uint8)


def log_static_scene(
    data,
    ep_slice: slice,
    robot_indices: list[int],
    axis_length: float,
) -> None:
    """写入不随时间变化的场景内容：整条轨迹、起点、终点。

    static=True 表示这些实体在整段 recording 中保持不变。
    Rerun 也会参考完整轨迹范围，自动给出较合适的 3D 视野。
    """
    # 声明世界坐标系：右手系，Z 轴向上（常见机器人基座约定）
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for robot_idx in robot_indices:
        # 取出该臂整段末端位置，形状约为 (T, 3)
        positions = np.asarray(data[f"robot{robot_idx}_eef_pos"][ep_slice])
        base = f"world/robot{robot_idx}"

        # 黄色折线：完整运动轨迹
        rr.log(
            f"{base}/full_trajectory",
            rr.LineStrips3D(
                [positions],
                colors=[[255, 200, 40]],
                radii=max(axis_length * 0.025, 0.001),
                labels=[f"Robot {robot_idx} trajectory"],
            ),
            static=True,
        )
        # 绿色点：起点
        rr.log(
            f"{base}/start",
            rr.Points3D(
                [positions[0]],
                colors=[[50, 220, 80]],
                radii=max(axis_length * 0.08, 0.004),
                labels=["start"],
            ),
            static=True,
        )
        # 粉色点：终点
        rr.log(
            f"{base}/end",
            rr.Points3D(
                [positions[-1]],
                colors=[[240, 60, 180]],
                radii=max(axis_length * 0.08, 0.004),
                labels=["end"],
            ),
            static=True,
        )


def log_dynamic_episode(
    data,
    ep_slice: slice,
    robot_indices: list[int],
    camera_keys: list[str],
    axis_length: float,
    invert_gripper: bool,
) -> None:
    """按时间逐步写入动态内容：当前末端坐标系、夹爪宽度、同步相机图像。

    核心映射：
      eef_pos[t]              → Points3D  origin（白点）
      eef_rot_axis_angle[t]   → Arrows3D  frame（XYZ 坐标轴）
      gripper_width[t]        → Boxes3D 两指 + Scalars 曲线
      camera*_rgb[t]          → Image
      timestamp[t]            → rr.set_time(...)
    """
    # 相对时间（秒）：没有 timestamp 时退化为 0,1,2,...
    timestamps = (
        np.asarray(data["timestamp"][ep_slice], dtype=np.float64)
        if "timestamp" in data
        else np.arange(ep_slice.stop - ep_slice.start, dtype=np.float64)
    )
    timestamps -= timestamps[0]

    # 预先切片，避免循环里反复读 zarr
    positions = {
        idx: np.asarray(data[f"robot{idx}_eef_pos"][ep_slice])
        for idx in robot_indices
    }
    rotations = {
        idx: np.asarray(data[f"robot{idx}_eef_rot_axis_angle"][ep_slice])
        for idx in robot_indices
    }
    gripper_widths = {}
    for idx in robot_indices:
        gripper_key = f"robot{idx}_gripper_width"
        if gripper_key in data:
            gripper_widths[idx] = np.asarray(data[gripper_key][ep_slice]).reshape(-1)

    for local_step, elapsed in enumerate(timestamps):
        # 两条时间线：帧序号 step，以及相对秒数 time
        # Viewer 底部可拖动其中任一条来回放
        rr.set_time("step", sequence=local_step)
        rr.set_time("time", duration=float(elapsed))

        for robot_idx in robot_indices:
            position = positions[robot_idx][local_step]  # (3,) 当前位置
            rotvec = rotations[robot_idx][local_step]    # (3,) 轴角姿态

            # 轴角 → 旋转矩阵；再把局部 XYZ 轴旋到世界坐标系
            # axis_vectors 的三列分别是世界系下的 X/Y/Z 轴方向，长度 = axis_length
            rotation_matrix = Rotation.from_rotvec(rotvec).as_matrix()
            axis_vectors = rotation_matrix * axis_length
            # 三个箭头共用同一个起点（末端位置）
            origins = np.repeat(position[None, :], 3, axis=0)
            base = f"world/robot{robot_idx}/eef"

            gripper_width = None
            gripper_opening = None
            if robot_idx in gripper_widths:
                gripper_width = float(gripper_widths[robot_idx][local_step])
                gripper_opening = stored_to_opening(gripper_width, invert_gripper)

            origin_label = None
            if gripper_width is not None and gripper_opening is not None:
                origin_label = [
                    f"open {gripper_opening * 1000:.1f} mm"
                ]

            # 白点：当前末端位置；有夹爪数据时标出当前宽度
            rr.log(
                f"{base}/origin",
                rr.Points3D(
                    [position],
                    colors=[[255, 255, 255]],
                    radii=max(axis_length * 0.06, 0.003),
                    labels=origin_label,
                ),
            )
            # 红绿蓝箭头：当前末端坐标系
            rr.log(
                f"{base}/frame",
                rr.Arrows3D(
                    origins=origins,
                    vectors=axis_vectors.T,  # (3,3)：三根轴的方向向量
                    colors=AXIS_COLORS,
                    radii=max(axis_length * 0.02, 0.001),
                    labels=["X", "Y", "Z"],
                ),
            )
            if gripper_width is not None and gripper_opening is not None:
                log_gripper(
                    base,
                    position,
                    rotvec,
                    gripper_opening,
                    gripper_width,
                    axis_length,
                )
                rr.log(
                    f"plots/robot{robot_idx}/gripper_width_stored",
                    rr.Scalars(gripper_width),
                )
                rr.log(
                    f"plots/robot{robot_idx}/gripper_opening",
                    rr.Scalars(gripper_opening),
                )

        # 相机图像与当前帧对齐；global_step 是整库中的绝对下标
        global_step = ep_slice.start + local_step
        for camera_key in camera_keys:
            rr.log(
                f"cameras/{camera_key}",
                rr.Image(image_as_uint8(data[camera_key][global_step])),
            )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="在 Rerun 中动态显示 Zarr episode 的末端轨迹、坐标系和夹爪宽度"
    )
    parser.add_argument(
        "--zarr_path",
        type=Path,
        default=Path("data/collect_demo/replay_buffer.zarr"),
        help="replay_buffer.zarr 路径",
    )
    parser.add_argument("--episode", type=int, default=0, help="要查看的 episode")
    parser.add_argument(
        "--axis_length",
        type=float,
        default=0.08,
        help="末端坐标轴显示长度，单位 m",
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="不加载相机图像，可减少内存和启动时间",
    )
    parser.add_argument(
        "--no_invert_gripper",
        action="store_true",
        help="不要把 gripper_width 取反。默认会取反：本数据集越大越夹紧",
    )
    parser.add_argument(
        "--save_rrd",
        type=Path,
        default=None,
        help="保存为 .rrd 文件而不直接启动 Viewer",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="启动 Web Viewer（SSH 远程推荐；配合本机端口转发）",
    )
    parser.add_argument(
        "--web_port",
        type=int,
        default=9090,
        help="Web Viewer HTTP 端口（默认 9090）",
    )
    parser.add_argument(
        "--grpc_port",
        type=int,
        default=9876,
        help="Rerun gRPC 端口（默认 9876）",
    )
    return parser.parse_args()


def main() -> None:
    """主流程：读 Zarr → 写 Rerun → 按模式启动 Viewer / 导出文件。"""
    args = parse_args()
    if not args.zarr_path.exists():
        raise FileNotFoundError(f"找不到 Zarr 数据：{args.zarr_path}")
    if args.axis_length <= 0:
        raise ValueError("--axis_length 必须大于 0")
    if args.save_rrd is not None and args.web:
        raise ValueError("--save_rrd 与 --web 不能同时使用")

    # ---------- 1) 打开 replay buffer ----------
    root = zarr.open(str(args.zarr_path), mode="r")
    if "data" not in root or "meta" not in root:
        raise ValueError("Zarr 中缺少 data 或 meta 分组")

    data = root["data"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if len(episode_ends) == 0:
        raise ValueError("Zarr 中没有 episode")

    # ---------- 2) 选定 episode，并确认必要字段 ----------
    ep_slice = episode_slice(episode_ends, args.episode)
    robot_indices = find_robot_indices(data)
    if not robot_indices:
        raise ValueError("没有找到 robot*_eef_pos")
    for robot_idx in robot_indices:
        rotation_key = f"robot{robot_idx}_eef_rot_axis_angle"
        if rotation_key not in data:
            raise ValueError(f"缺少 {rotation_key}，无法显示末端坐标系")

    camera_keys = [] if args.no_images else find_camera_keys(data)
    # 仅在“本机直接看”时 spawn 原生 Viewer；导出 / Web 模式都不弹本地窗口
    spawn_native = (args.save_rrd is None) and (not args.web)

    # ---------- 3) 决定输出目标 ----------
    # SSH / Web 模式：先完整写入 .rrd，再用 CLI 托管，避免浏览器空页面。
    rrd_path: Path | None = None
    temp_rrd: Path | None = None
    if args.save_rrd is not None:
        rrd_path = args.save_rrd
        rrd_path.parent.mkdir(parents=True, exist_ok=True)
    elif args.web:
        # Web 模式用临时文件承载 recording，进程退出时清理
        temp_rrd = Path(tempfile.gettempdir()) / (
            f"arx5_episode_{args.episode}_{os.getpid()}.rrd"
        )
        rrd_path = temp_rrd

    # ---------- 4) 初始化 Rerun，并写入场景 ----------
    rr.init(
        "ARX5 Zarr Trajectory",
        recording_id=f"{args.zarr_path.resolve()}-episode-{args.episode}",
        spawn=spawn_native,
    )
    if rrd_path is not None:
        # 之后的 rr.log 会写入该 .rrd 文件
        rr.save(rrd_path)

    invert_gripper = not args.no_invert_gripper
    log_static_scene(data, ep_slice, robot_indices, args.axis_length)
    log_dynamic_episode(
        data,
        ep_slice,
        robot_indices,
        camera_keys,
        args.axis_length,
        invert_gripper,
    )
    if rrd_path is not None:
        # 关闭写入，确保 .rrd 完整后再交给 CLI / scp
        rr.disconnect()

    n_grippers = sum(
        1 for idx in robot_indices if f"robot{idx}_gripper_width" in data
    )
    print(
        f"已载入 episode {args.episode}: "
        f"{ep_slice.stop - ep_slice.start} steps, "
        f"{len(robot_indices)} robot(s), {n_grippers} gripper(s), "
        f"{len(camera_keys)} camera(s)"
    )
    if n_grippers:
        polarity = (
            "已取反显示物理开口（zarr 原值越大越夹紧）"
            if invert_gripper
            else "按 zarr 原值显示（未取反）"
        )
        print(f"夹爪: {polarity}")

    # ---------- 5) 按模式收尾 ----------
    if args.save_rrd is not None:
        # 仅导出，不启动任何 Viewer
        print(f"Rerun 数据已保存到：{rrd_path}")
        print("可用 scp 拷到本机后执行: rerun <file.rrd>")
    elif args.web:
        # Web Viewer：用 CLI 托管刚写好的 .rrd
        assert rrd_path is not None
        if not rrd_path.exists() or rrd_path.stat().st_size == 0:
            raise RuntimeError(f"生成的 .rrd 无效：{rrd_path}")

        # 浏览器必须通过 ?url=... 连到 gRPC，否则只有空壳页面
        grpc_url = f"rerun+http://127.0.0.1:{args.grpc_port}/proxy"
        browser_url = (
            f"http://127.0.0.1:{args.web_port}/?url={quote(grpc_url, safe='')}"
        )
        cmd = [
            "rerun",
            str(rrd_path),
            "--web-viewer",
            "--web-viewer-port",
            str(args.web_port),   # HTTP 页面端口
            "--port",
            str(args.grpc_port),  # 数据通道端口
            "--bind",
            "127.0.0.1",
        ]
        print("正在启动 Web Viewer ...")
        print("  " + " ".join(cmd))
        proc = subprocess.Popen(cmd)

        def _cleanup() -> None:
            """退出时关掉 rerun 子进程，并删除临时 .rrd。"""
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if temp_rrd is not None and temp_rrd.exists():
                temp_rrd.unlink(missing_ok=True)

        atexit.register(_cleanup)

        print("Web Viewer 已启动（适合 SSH）。")
        print("重要：必须同时转发 HTTP 和 gRPC 两个端口，否则只有空界面。")
        print("在本机另开终端:")
        print(
            f"  ssh -L {args.web_port}:127.0.0.1:{args.web_port} "
            f"-L {args.grpc_port}:127.0.0.1:{args.grpc_port} <user>@<server>"
        )
        print("然后本机浏览器打开下面这个带连接参数的地址:")
        print(f"  {browser_url}")
        print("打开后应能看到 world/robot0 轨迹；底部拖动 time/step 回放。")
        print("按 Ctrl+C 结束服务。")
        try:
            # 阻塞保活：进程退出后 Web Viewer 也会停
            while proc.poll() is None:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("已停止 Web Viewer。")
            _cleanup()
    else:
        # 本机原生 Viewer 已由 rr.init(spawn=True) 拉起
        print("请在 Rerun Viewer 底部拖动 time 或 step 时间轴进行回放。")


if __name__ == "__main__":
    main()
