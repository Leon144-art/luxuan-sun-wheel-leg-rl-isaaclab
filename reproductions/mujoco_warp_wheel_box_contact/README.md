# MuJoCo Warp 轮子与箱体接触异常复现

本目录包含一个可独立运行的 CPU/GPU 接触差异复现。问题发生在圆柱形机器人轮子与箱体台阶接触时，不依赖训练环境、PPO、奖励函数、策略 checkpoint 或多环境并行。

## 测试环境

参考结果使用以下环境生成：

- Python 3.12；
- MuJoCo 3.8.1；
- MuJoCo Warp 3.8.1；
- Warp 1.14.0；
- NumPy 2.5.2；
- NVIDIA CUDA GPU。

建议在临时虚拟环境中安装固定版本的物理仿真依赖：

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 运行方法

```bash
.venv/bin/python reproduce.py --output result.json
```

如需检查 MuJoCo Warp 3.8.1 是否出现本项目记录的完整故障特征，可运行：

```bash
.venv/bin/python reproduce.py \
  --output result.json \
  --expect-known-bug
```

`--expect-known-bug` 只用于检查3.8.1版本的已知故障特征。如果后端已经修复，该检查应当不再通过；普通复现命令仍会正常完成，并输出新的 CPU/GPU 对照结果。

## 对照项目与参考结果

程序分别创建全新的CPU和GPU数据，并执行以下4组单环境 `forward` 对照：

| 模型与坐标 | CPU MuJoCo 3.8.1 | GPU MuJoCo Warp 3.8.1 |
| --- | --- | --- |
| 完整碰撞模型，捕获时的世界坐标 | 4个位置合理的轮—箱体接触，`qacc` 有限 | 1个距轮心约1951.56m的接触，`qacc` 非有限 |
| 完整碰撞模型，机器人与箱体共同平移到原点附近 | 4个位置合理的接触，`qacc` 有限 | 没有轮—箱体接触，`qacc` 有限 |
| 单圆柱与单箱体，捕获时的世界坐标 | 1个位置合理的接触，`qacc` 有限 | 1个位置合理的接触，`qacc` 有限 |
| 单圆柱与单箱体，平移到原点附近 | 1个位置合理的接触，`qacc` 有限 | 1个位置合理的接触，`qacc` 有限 |

完整原始输出保存在 `reference_results.json`。共同平移不会改变两个碰撞体的相对几何关系，因此GPU结果由“错误接触”变成“漏接触”仍属于计算差异，不能视为修复。

## 文件说明

- `hopper_collision.xml` 保留原机器人的完整运动链、质量与惯量、底盘和腿部碰撞体以及圆柱形轮子碰撞体。视觉网格已替换为不参与碰撞的占位几何体，在无需STL资源的情况下保留原几何体顺序。
- `captured_state.json` 只包含复现所需的13维 `qpos`、箱体位置与尺寸、共同平移量以及求解器设置。
- `reproduce.py` 为每组测试分别创建新的CPU与GPU数据，并在转换回 `MjData` 前直接读取GPU接触数组。
- `reference_results.json` 是程序生成的完整参考输出。

## 结论与边界

该复现证明，异常不需要学习策略、优化器、奖励实现、MjLab环境循环或批量并行仿真即可出现。它同时表明：如果从CPU正运动学取得轮子姿态，再将场景简化成单圆柱和单箱体，当前条件下不足以复现该异常。

目前还不能确定剩余触发条件究竟来自：

- 完整运动链上的GPU正运动学；
- GJK/EPA数值退化；
- 坐标空间处理错误；
- 其他仅在完整模型中出现的交互。

MuJoCo Warp 上游还分别记录了 GJK/EPA 对共同世界坐标平移敏感，以及圆柱—箱体接触流形与 CPU MuJoCo 不一致的问题：

- <https://github.com/google-deepmind/mujoco_warp/issues/1546>
- <https://github.com/google-deepmind/mujoco_warp/issues/1555>

在加入专门的台阶策略网络之前，一个较早的完整训练也记录过3次被隔离的非有限物理状态。使用该旧配置和固定策略进行短测时，还在高度图斜坡上发现过偏离轮子的GPU接触点。这些观察没有作为可执行文件放入本目录，因为它们依赖私有任务代码和策略checkpoint，而且尚不能证明它们与本独立BOX复现具有相同根因。
