# soundsculpt — 基于小波包分解的声音雕刻

把录音按选定小波包基和分解树变成多尺度系数，在指定节点和时间范围上施加
增益，再逆变换回音频。这是真正的小波包表示，不是按频带增益的均衡器：
每个节点是一组带通尺度函数的系数，频带相互交叠，编辑会沿基函数支持
向邻近时频位置传播。

## 安装

```bash
python3 -m pip install --user numpy PyWavelets pytest
```

## 两种笔迹（操作语义不同）

| 笔迹 | 记忆内容 | 重选基后的行为 |
|------|----------|----------------|
| `band` 圈选频带 | 物理频率区间 (Hz) | 始终覆盖同一物理频带，覆盖节点可变化 |
| `subtree` 压低子树 | 分解树节点路径 | 跟随该节点的后代；节点在新树中不存在时进入 unresolved，必须人工 retarget |

同一基下，对同一节点的全时长 band 圈选与 subtree 压低产生**逐样本相同**
的输出（测试 `test_band_and_subtree_strokes_identical`，误差 < 1e-12）。

## 命令行

```bash
# 建工程（内置 demo/chirp/impulse/boundary/noise 信号或 --signal wav）
python -m soundsculpt.cli init p.json --signal demo --wavelet db4 --level 5 --mode periodization

# 检查树：标称频带、系数步长、实测基函数支持、边缘系数
python -m soundsculpt.cli tree p.json --support
python -m soundsculpt.cli bands p.json
python -m soundsculpt.cli leakage p.json ad      # 非砖墙证据：能量跨标称带分布

# 两种笔迹（--t0/--t1 给时间窗，省略 t1 表示到文件尾）
python -m soundsculpt.cli stroke-band p.json b1 --node ad --gain 0.25
python -m soundsculpt.cli stroke-subtree p.json s1 ad --gain 0.25

# 渲染与无编辑往返
python -m soundsculpt.cli render p.json edited.wav

# 重选基（换小波/深度/延拓），无法映射的子树笔划会被报告
python -m soundsculpt.cli rebase p.json --wavelet sym6 --level 6
python -m soundsculpt.cli retarget p.json s1 ada   # 人工重定目标

# 按节点身份施加系数编辑（时间窗，节点必须是当前叶），并撤销
python -m soundsculpt.cli edit p.json adda --gain 0.5 --t0 0.1 --t1 0.3 --tag e1
python -m soundsculpt.cli edits p.json          # 列出节点身份 + 树版本
python -m soundsculpt.cli undo p.json --tag e1  # 按节点身份撤销

# 一高一低探测音：测量重开后的实际增益（FFT 读数，含邻带泄漏）
python -m soundsculpt.cli probe p.json --low 300 --high 7000

# 单节点贡献试听
python -m soundsculpt.cli audition p.json ad node_ad.wav

# 数值验证电池（脉冲/边界/变频音/窄带噪声/demo 五类信号）
python -m soundsculpt.cli verify report.json

# 换树 + 边界短片的前后差分证据（重建误差与节点归属）
python -m soundsculpt.cli diff report_diff.json --signal chirp
```

## 浏览器界面

```bash
python -m soundsculpt.web p.json --port 8000
```

启动后打开 `http://127.0.0.1:8000/`。界面包含：

- **左上分解树**：可展开、查看每节点标称频带/实测支持/边缘系数/泄漏剖面、
  试听单节点、局部细分（subdivide）。
- **频谱编辑窗口**（右上）：绘制当前渲染结果的时频能量图与波形；点击某个
  时频单元即选中一个当前叶节点 + 时间窗（绿色框）。`scale depth` 滑块切换
  分解尺度，「switch scale」通过 WPT 内核重选深度并把编辑沿祖先路径迁移；
  「apply edit to picked node」把增益写入所选节点并重绘频谱与波形；
  「undo」按节点身份撤销。失败（如选中的不是叶节点）会在消息区显示。
- 两种笔划、重选基并对 unresolved 笔划点选候选节点重定目标、跑双探测音、
  下载渲染 WAV。

界面的所有编辑都由 `soundsculpt` 的小波内核驱动：`/api/edit` 调用
`Project.add_node_edit`（按 `NodeId` 绑定），`/api/scale` 调用 `Project.rebase`
（迁移编辑），渲染始终固定到显式的树版本，不会读取隐式/默认的树。

## 分解树接口（重构边界）

重构把原先缠在一起的 `BasisSpec` / `WPTEngine` 拆成三块，彼此以明确接口为界：

- **基函数配置** `BasisSpec`（`wavelet`/`max_level`/`mode`/`leaves`）——保存
  格式不变，旧工程 JSON 可直接加载。
- **分解树** `DecompositionTree`（`soundsculpt/tree.py`）——稀疏保存叶切分
  （**不**预展平到最深一层），提供 `TreeSpec`、稳定的 `NodeId`、内容哈希的
  `TreeVersion`。逆变换 `WPTEngine.reconstruct(..., tree_version=...)` 只接受
  显式版本，版本不符抛 `StaleTreeVersion`。
- **编辑系数状态** `EditState`（`soundsculpt/edits.py`）——每条记录以 `NodeId`
  为键，含秒级时间窗意图。撤销与保存读同一身份。

树版本变化时的编辑迁移规则（`EditState.migrate`）：

- 旧节点在新树中是某些叶的**祖先**（细分）→ 沿祖先路径重新映射到这些叶；
- 新树只有旧节点的**粗祖先**（合并/降深度）→ **拒绝**该编辑并回滚，避免把
  细编辑写到从未触碰的旁支（有测试断言旁支系数未被回滚）；
- 无任何可比节点 → 拒绝。

「保留原树供非破坏求值」：`Project.evaluate_tree(tree, state)` 用一对冻结的
`(tree, state)` 求值，互不修改；版本不匹配会抛 `StaleTreeVersion`。

## 关键设计与诚实的限制

- **可逆重建**：所有延拓模式下无编辑往返误差 ~1e-15（浮点），PCM 量化后
  为 0 LSB。混合叶基通过把粗叶系数前向下推到最大深度再重构实现。
- **父子不重复计入**：`reconstruct` 校验叶基互不构成前缀且完整覆盖；
  同一节点与其后代不可能同时进入重构基。
- **不是砖墙频带**：`leakage_profile` 用白噪声实测节点能量跨标称带的分布，
  例如 db4/L5 的 `ad` 节点仅约 73% 能量落在自身标称带。界面和报告都展示
  这一点，不声称理想隔离。
- **局部编辑会外溢**：时间窗编辑的残差范围由实测基函数支持决定，脉冲信号
  上残差集中在脉冲附近，变频音上可观察到振铃超出选区。验证覆盖这一传播，
  而非只看选区。
- **边缘延拓**：对称/周期类模式在边界产生冗余帧（系数能量比 >1），
  `periodization`/`zero` 保 Parseval 能量；`tree --support` 给出受边缘
  影响的系数个数和并集支持。
- **稀疏信号**：脉冲/边界信号在某节点+时间窗内可能没有系数能量，此时编辑
  无作用是正确行为，验证电池将其标记为 `skipped_sparse` 而非失败。

## 目录

- `soundsculpt/` 真实变换与应用代码
  - `paths.py` 纯路径/频带几何（无状态）
  - `tree.py` 显式、带版本的分解树（新边界）
  - `edits.py` 按节点身份的系数编辑状态、迁移与撤销
  - `wpt.py` 变换引擎（逆变换固定到显式树版本，不再预展平）
  - `basis.py` 基配置、笔划、`Project`（撤销/保存共享节点身份）
  - `verification.py` / `web.py` / `cli.py`
- `tests/` 46 个 pytest 用例（含 `test_tree_interface.py` 的接口契约）
- `outputs/` 处理后的 WAV、探测音、节点试听、传播与验证报告、工程 JSON
