"""边训练边演示：自动跟随训练产出的最新检查点，无需重启 Isaac Sim。

原理：`OnPolicyRunner.get_inference_policy()` 返回的是 `actor_critic.act_inference`
这个**绑定方法**，而 `load()` 是通过 `load_state_dict()` **原地修改**同一个
actor_critic 对象。因此只要在渲染循环里重新调用一次 `runner.load(新检查点)`，
已经在跑的策略对象就会立刻用上新权重 —— 不用重启 Isaac Sim。

用法（**不要加 --headless**，否则看不到窗口）::

    # 自动监视"最近被修改过的 run 目录"，每 20 秒检查一次新检查点
    python locowheeledlegged/scripts/watch.py \
        --task Isaac-LocomotionGo2W-Play-v1 --num_envs=1

    # 指定要监视的 run 目录
    python locowheeledlegged/scripts/watch.py \
        --task Isaac-LocomotionGo2W-Play-v1 --num_envs=1 \
        --watch_dir logs/rsl_rl/locowheeledlegged_go2w/2026-09-17_11-34-19

建议配合 2048 环境以内的训练同时运行（4096 环境的训练会占满显存，见 README）。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher
import cli_args

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="自动跟随最新检查点的演示脚本")
parser.add_argument("--num_envs", type=int, default=None, help="环境数量(演示建议 1~20)")
parser.add_argument("--task", type=str, default="Isaac-LocomotionGo2W-Play-v1", help="任务名")
parser.add_argument("--watch_dir", type=str, default=None,
                    help="要监视的 run 目录；不填则自动选最近修改过的")
parser.add_argument("--reload_every", type=float, default=20.0,
                    help="每隔多少秒检查一次新检查点(默认 20)")
parser.add_argument("--keep_training_commands", action="store_true", default=False,
                    help="保留训练时的指令分布(默认关闭 bang-bang / 强制站立指令，演示更容易看懂)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 其余导入必须在 App 启动之后
# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

# 把项目根目录加入 sys.path，以便 import locowheeledlegged
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from locowheeledlegged import *  # noqa: E402,F401,F403
from loco_rl.runners import OnPolicyRunner  # noqa: E402

LOG_ROOT = os.path.join(_project_root, "logs", "rsl_rl", "locowheeledlegged_go2w")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def list_runs_newest_first() -> list[str]:
    """列出所有 run 目录，按最后修改时间倒序。"""
    if not os.path.isdir(LOG_ROOT):
        return []
    runs = [os.path.join(LOG_ROOT, d) for d in os.listdir(LOG_ROOT)
            if os.path.isdir(os.path.join(LOG_ROOT, d))]
    return sorted(runs, key=os.path.getmtime, reverse=True)


def newest_checkpoint(run_dir: str | None) -> str | None:
    """返回 run 目录里编号最大的 model_<n>.pt。"""
    if not run_dir or not os.path.isdir(run_dir):
        return None
    best_path, best_iter = None, -1
    for fname in os.listdir(run_dir):
        m = re.fullmatch(r"model_(\d+)\.pt", fname)
        if not m:
            continue
        it = int(m.group(1))
        if it > best_iter:
            best_path, best_iter = os.path.join(run_dir, fname), it
    return best_path


def describe(path: str | None) -> str:
    if not path:
        return "<无检查点>"
    m = re.search(r"model_(\d+)\.pt", path)
    run = os.path.basename(os.path.dirname(path))
    try:
        size = os.path.getsize(path) / 1048576
        size_s = f"{size:.1f}MB"
    except OSError:
        size_s = "?"
    return f"iter={m.group(1) if m else '?'}  {run}  ({size_s})"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # --- 让相机跟着机器人，而不是死盯世界原点 ---
    # 地形有 10x20 块 8m 瓦片，机器人会被随机分配到某块上，
    # 默认 origin_type="world" 时相机看世界原点，根本看不到机器人。
    try:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.eye = (3.0, 3.0, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.4)
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.asset_name = "robot"
        print("[WATCH] 相机已设为跟随机器人 (origin_type=asset_root)")
    except Exception as exc:  # noqa: BLE001
        print(f"[WATCH] 相机设置失败，沿用配置默认值: {exc}")

    # --- 演示友好的指令分布 ---
    # 训练时 bang_bang_envs=0.05、play 配置里是 0.5（极限指令），
    # rel_standing_envs=0.1（10% 环境强制站立）。单台演示时这两种都会让人
    # 误以为"策略不会动"。默认关掉，想看原版分布就加 --keep_training_commands。
    if not args_cli.keep_training_commands:
        try:
            env_cfg.commands.base_velocity.bang_bang_envs = 0.0
            env_cfg.commands.base_velocity.rel_standing_envs = 0.0
            print("[WATCH] 已关闭极限指令(bang_bang)与强制站立指令")
        except Exception as exc:  # noqa: BLE001
            print(f"[WATCH] 指令设置失败: {exc}")

    # --- 建环境 ---
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # --- 确定要监视的 run ---
    run_dir = args_cli.watch_dir
    if run_dir and not os.path.isabs(run_dir):
        run_dir = os.path.join(_project_root, run_dir)
    if not run_dir:
        runs = list_runs_newest_first()
        run_dir = runs[0] if runs else None

    ckpt = newest_checkpoint(run_dir)
    print(f"[WATCH] 监视目录 : {run_dir}")
    print(f"[WATCH] 初始检查点: {describe(ckpt)}")

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=os.path.dirname(ckpt) if ckpt else None,
        device=agent_cfg.device,
    )
    if ckpt:
        runner.load(ckpt)
    else:
        print("[WATCH] 暂无检查点，先用随机初始策略演示")

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.get_observations()

    current_ckpt = ckpt
    last_check = time.time()
    switched = 0
    print(f"[WATCH] 就绪。每 {args_cli.reload_every:.0f} 秒检查一次新检查点，Ctrl+C 退出。")

    while simulation_app.is_running():
        # --- 定期检查是否有新检查点 ---
        if time.time() - last_check >= args_cli.reload_every:
            last_check = time.time()
            if args_cli.watch_dir is None:
                runs = list_runs_newest_first()
                run_dir = runs[0] if runs else run_dir
            cand = newest_checkpoint(run_dir)
            if cand and os.path.abspath(cand) != os.path.abspath(current_ckpt or ""):
                try:
                    runner.load(cand)
                    # 权重是原地更新的，但重新取一次更保险
                    policy = runner.get_inference_policy(device=env.unwrapped.device)
                    current_ckpt = cand
                    switched += 1
                    print(f"[WATCH] >>> 第 {switched} 次切换 -> {describe(cand)}")
                except Exception as exc:  # noqa: BLE001
                    # 检查点可能正在写入，下一轮再试
                    print(f"[WATCH] 加载 {os.path.basename(cand)} 失败，稍后重试: {exc}")

        # --- 推理 + 步进 ---
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()
    print(f"[WATCH] 退出。共切换检查点 {switched} 次。")


if __name__ == "__main__":
    main()
    simulation_app.close()
