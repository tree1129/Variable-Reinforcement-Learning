# tree_Reinforcement_Learning

Case 4–7 三面精准投放任务的**独立强化学习工程**。

> 本目录完全独立，不导入、不修改仓库中的 `tree_center_brain`、Wall-X、OpenPI 或其他训练文件。默认的 `ToyBackend` 只用于接口 smoke test，不能代表真实机械臂性能，也不能直接连接真机。

## 目标任务

| Case | 正面孔 | 右侧面孔 | 左侧面孔 | 不投入 | 目标位置 |
|---|---|---|---|---|---|
| 4 | 三角形 | 方形 | 球形 | 梯形 | 盒体中中；三角形左上、方形右上、梯形左下、球形右下 |
| 5 | 方形 | 梯形 | 三角形 | 球形 | 盒体左中；三角形中上、方形右上、梯形中下、球形右下 |
| 6 | 梯形 | 球形 | 方形 | 三角形 | 盒体右中；三角形左上、方形中上、梯形左下、球形中下 |
| 7 | 球形 | 三角形 | 梯形 | 方形 | 盒体中中；三角形左中、方形中上、梯形右中、球形中下 |

每个 Case 的最终比赛分数独立于训练 reward：

```text
三个目标正确插入：每个 3 分
三个目标全部完成后机械臂安全收回：1 分
满分：10 分
```

## 架构

```text
相机/RGB-D/位姿估计
        ↓
Case 4–7 task condition + 短指令
        ↓
SAC/PPO policy（输出归一化末端增量）
        ↓
动作裁剪 + 工作空间/碰撞/力限制 + IK/阻抗控制
        ↓
仿真器或受保护的机器人 adapter
```

策略动作固定为 7 维：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

动作是归一化的 `[-1, 1]`，后端必须负责把它转换为真实单位并执行安全限制。

## 安装

只安装这个独立项目的依赖：

```bash
cd tree_Reinforcement_Learning
python3 -m pip install -e '.[rl]'
```

如果只想检查任务定义和 Prompt，不需要安装 Gym/SB3：

```bash
PYTHONPATH=src python scripts/validate_tasks.py
PYTHONPATH=src python scripts/train.py --dry-run
```

## 训练

### 1. 接口 smoke test

```bash
PYTHONPATH=src python scripts/train.py --backend toy --steps 10000
```

ToyBackend 是运动学简化环境，只验证 Gym/SAC 接口，不用于真实性能结论。

### 2. 接入物理仿真器

新建一个**外部** adapter，例如：

```python
# my_sim/adapter.py

def make_backend():
    return MyMuJoCoBackend()
```

adapter 必须实现：

```python
class Backend:
    def reset(self, case_id: int, seed: int | None = None) -> None: ...
    def get_observation(self) -> dict: ...
    def apply_action(self, action) -> None: ...
    def close(self) -> None: ...
```

然后运行：

```bash
PYTHONPATH=src:/path/to/my_sim python scripts/train.py \
  --backend custom \
  --backend-factory my_sim.adapter:make_backend \
  --steps 300000
```

### 3. 训练建议

1. 每个 Case 先采集 30–50 条成功示教轨迹；
2. 先行为克隆 warm-start，再用 SAC/AWAC/IQL 微调；
3. 训练时随机化积木位置、盒体位置、姿态、相机视角和光照；
4. 按完整 episode 划分 train/validation/test，不能按帧随机划分；
5. 先单目标插入，再三个目标，最后四个 Case 混合训练；
6. 真机只允许低速、小步长、限力、可急停的 safety wrapper 接管。

## 后端观测协议

`get_observation()` 必须返回以下字段：

```text
object_positions:      (4, 3), 单位 m，顺序 triangle/square/trapezoid/sphere
object_orientations:   (4, 3)，统一使用 axis-angle 或等价 3-vector
box_pose:              (6,), 位置 + 姿态
hole_poses:            (3, 6)，顺序 front/right/left，位置 + 姿态
ee_pose:                (6,), 末端位置 + 姿态
gripper:               标量，建议 [-1, 1]
inserted:              (3,), bool，顺序为当前 Case 的 targets
retracted:             bool
grasped:               bool
grasp_success:         bool，仅在当前 transition 首次抓取成功时为 true
forbidden_contact:     bool
box_displacement_m:    标量
contact_force_n:       标量
```

## 安全边界

`reward.py` 中的安全终止不是硬件安全系统。接入真实机械臂前必须额外实现：

- 关节角、关节速度、加速度限制；
- 末端工作空间限制；
- 碰撞检测与急停；
- 力/力矩阈值；
- 盒体位移检测；
- 动作超时和通信断连保护；
- 人工确认和低速 dry-run。

## A100 部署（SAC 主算法）

项目提供了独立部署脚本 `scripts/deploy_a100.sh`。它默认使用 SSH 配置中的 `a100` 主机，远端目录为 `/root/tree_Reinforcement_Learning`，会：

1. 上传本目录（排除 macOS `._*` 元数据、缓存和训练输出）；
2. 创建远程 `.venv`；
3. 安装本项目及 `gymnasium/stable-baselines3`；
4. 运行单元测试、Case 校验和 SAC dry-run。

默认只部署和验证，不会在没有真实仿真 Backend 的情况下盲目启动训练：

```bash
cd tree_Reinforcement_Learning
bash scripts/deploy_a100.sh
```

远端 A100 可用、且已经安装了物理仿真器 adapter 后，使用 SAC 启动训练：

```bash
bash scripts/deploy_a100.sh \
  --train \
  --backend custom \
  --backend-factory my_sim.adapter:make_backend \
  --steps 300000 \
  --device cuda
```

训练日志位于远端：

```text
/root/tree_Reinforcement_Learning/outputs/case4_7_sac_a100/train.log
```

若仅需检查 SB3/ToyBackend 接口，可明确指定：

```bash
bash scripts/deploy_a100.sh --train --backend toy --steps 10000
```

但 ToyBackend 是非物理 smoke test，不能用于证明真实机械臂性能。当前本机尝试连接 SSH `a100` 时远端 banner 超时，因此尚未宣称远程部署或训练已经成功；网络/机器人隧道恢复后重新执行上述命令即可。

### 4. SSH A100 上使用 SAC + MuJoCo

当前独立工程已支持无显示（headless）MuJoCo adapter，默认读取：

```text
/root/thu_robot_sim/project/artifacts/scene_twin/maps/v0010_operable_workcell/operable_workcell.xml
```

先部署、安装依赖并验证 Case 4–7：

```bash
cd tree_Reinforcement_Learning
bash scripts/deploy_a100.sh --backend mujoco
```

验证通过后启动 A100 上的 SAC 训练（默认远端目录 `/root/tree_Reinforcement_Learning`）：

```bash
bash scripts/deploy_a100.sh \
  --train \
  --backend mujoco \
  --device cuda \
  --steps 300000
```

训练会在 SSH 断开后继续运行。查看日志和进程：

```bash
ssh a100 'tail -f /root/tree_Reinforcement_Learning/outputs/case4_7_sac_a100/train.log'
ssh a100 'pgrep -af "scripts/train.py.*case4_7_sac_a100"'
```

模型和 TensorBoard 日志分别写入：

```text
/root/tree_Reinforcement_Learning/outputs/case4_7_sac_a100/sac_case4_7.zip
/root/tree_Reinforcement_Learning/outputs/case4_7_sac_a100/tensorboard/
```

`validate_mujoco_backend.py` 只验证 headless backend 的 reset、step、观测协议和抓取/释放状态机；它不是对真实机械臂成功率的保证。真机前仍需独立的关节/速度/力/碰撞/急停和通信 watchdog 安全层。
