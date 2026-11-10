# RBF-FD 频响学习：三类方形板与模态有理增广

本工程保留统一的频响学习流程：预装配 RBF-FD 频域算子，在约束坐标中训练全阶网络或 POD 坐标网络，并可选地加入共享的模态有理分支与在线残差矫正。

## 三个算例

| 算例键 | 结构 | 默认节点设置 |
|---|---|---|
| `flat` | 四边简支实心方形 Mindlin 板 | `31 × 31` 节点，`--square-ndiv 30` |
| `hollow` | 四边简支、中心圆孔方形 Mindlin 板 | 圆孔直径/边长为 `1/4`；4 个扫掠分区；扫掠方向 30 点 |
| `stiffened` | 四边简支方形板与嵌入式 Timoshenko 加筋梁 | 与实心方板相同的基板节点 |

环形板已从默认算例、批处理、汇总和论文绘图入口中移除。旧环形板底层函数仅作为历史兼容代码保留，不参与当前实验。

### 中空方形板布点

中心圆孔半径为

\[
r_h=\frac{1}{8}L,
\]

即圆孔直径为方形板边长的 \(1/4\)。圆周按四个等长四分之一圆弧划分，每个圆弧分别映射到上、右、下、左外边界。每个分区采用

\[
\mathbf{x}(t,s)=(1-s)\mathbf{x}_{\rm hole}(t)+s\mathbf{x}_{\rm edge}(t),
\qquad t\in[-1,1],\ s\in[0,1],
\]

进行扫掠。默认 `--hollow-sweep-points 15`、`--hollow-tangent-points-per-patch 16`；四个映射分区合并后为 900 个物理节点，与 31×31 实心/加筋方板保持相近自由度。四个分区的公共界面节点会自动合并。外边界施加与另外两个方板相同的简支约束，圆孔边界由弱式自然形成自由边界。

## POD + 模态有理增广

启用 `--modal-resonance` 后，全阶和 POD 两条路线使用同一组约束模态、结构阻尼、共振频率和模态广义力。POD 表示改为

\[
\mathbf{u}(\omega)
=
\boldsymbol{\Phi}_{\rm POD}\mathbf{a}_{\rm NN}(\omega)
+
\mathbf{V}_{\rm modal}\mathbf{r}_{\rm rational}(\omega).
\]

实现中，先从高保真快照中减去共享模态分支：

\[
\mathbf{u}_{\rm bg}(\omega)
=
\mathbf{u}_{\rm ref}(\omega)-\mathbf{u}_{\rm modal}(\omega),
\]

再用背景快照构造 POD 基底并训练坐标网络。模态分支采用当前结构阻尼形式

\[
\mathbf{u}_{\rm modal}(\omega)
=
\sum_r \boldsymbol{\phi}_r
\frac{\boldsymbol{\phi}_r^{\mathsf T}\mathbf{f}}
{(1+i\eta)\lambda_r-\omega^2}.
\]

物理残差、预测和 POD 在线坐标矫正均对完整响应

\[
\mathbf{u}_{\rm POD-bg}+\mathbf{u}_{\rm modal}
\]

进行计算。共振窗口可使用与全阶路线相同的训练掩码排除。因此，当前 `pod_*` 结果在启用该选项时已经是“POD 背景 + 模态有理分支”，不再是普通 POD 直接拟合。

## 快速测试

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
& ".\.venv\Scripts\python.exe" -m pytest .\tests\test_full_order_workflow.py -q
```

## 单算例运行

```powershell
& ".\.venv\Scripts\python.exe" .\run_frequency_validation.py `
  --case hollow `
  --epochs 1200 `
  --n-labeled 48 `
  --n-physics 160 `
  --frequency-min 0 --frequency-max 200 `
  --modal-resonance --modal-modes 8 `
  --compare-pod --pod-modes 8 --pod-gradient-balance `
  --refine-steps 500 --pod-refine-steps 500 `
  --hollow-sweep-points 15 `
  --hollow-tangent-points-per-patch 16 `
  --hollow-stencil 20 `
  --hollow-poly-degree 2 `
  --hole-diameter-ratio 0.25 `
  --sweep-frequency-min 0 --sweep-frequency-max 200 --sweep-points 801 `
  --device cuda `
  --out .\results\unified_square_cases_modal_sweep_cuda
```

将 `--case` 改为 `flat` 或 `stiffened` 可运行其余算例。频率样本数量已保留为命令行参数，建议正式共振分析使用比旧版 24/80 更密的标签与物理频率集，并使用至少 801 个后处理扫频点检查窄峰。

## 三算例批处理

```powershell
& ".\.venv\Scripts\python.exe" .\run_all_full_order_validations.py `
  --epochs 1200 --device cuda `
  --out .\results\unified_square_cases_modal_sweep_cuda
```

批处理顺序为 `flat`、`hollow`、`stiffened`。

## 结果汇总与绘图

```powershell
& ".\.venv\Scripts\python.exe" .\summarize_frequency_sweeps.py `
  --root .\results\unified_square_cases_modal_sweep_cuda

& ".\.venv\Scripts\python.exe" .\generate_updated_manuscript_figures.py `
  --results-root .\results\unified_square_cases_modal_sweep_cuda `
  --sweep-root .\results\unified_square_cases_modal_sweep_cuda `
  --out .\results\unified_square_cases_modal_sweep_figures
```

中空方板绘图会自动屏蔽圆孔内部三角形，并同时绘制方形外边界和圆形孔边界。

## 八种配置

| 表示 | 训练目标 | 在线矫正 | 文件标识 |
|---|---|---|---|
| 全阶背景 + 模态分支 | data-only | 无 | `full_data_only` |
| 全阶背景 + 模态分支 | data-only | 有 | `full_data_only_refined` |
| 全阶背景 + 模态分支 | physics-constrained | 无 | `full_physics` |
| 全阶背景 + 模态分支 | physics-constrained | 有 | `full_physics_refined` |
| POD 背景 + 模态分支 | data-only | 无 | `pod_data_only` |
| POD 背景 + 模态分支 | data-only | 有 | `pod_data_only_refined` |
| POD 背景 + 模态分支 | physics-constrained | 无 | `pod_physics` |
| POD 背景 + 模态分支 | physics-constrained | 有 | `pod_physics_refined` |

未启用 `--modal-resonance` 时，接口自动退化为原全阶或普通 POD 模型。

## 关键输出

每个算例目录包含训练历史、验证指标、完整扫频响应与共享模态信息。POD 增广额外保存：

```text
pod_modal_natural_frequencies_hz.npy
pod_seed_training_mask.npy
pod_physics_training_mask.npy
```

全阶分支保存：

```text
modal_natural_frequencies_hz.npy
modal_resonance_frequencies_hz.npy
labeled_training_mask.npy
physics_training_mask.npy
```

## 代码结构

```text
rbf_fd_full_order/
├─ core/
│  ├─ rbf_fd_operator.py       # 三类方形板、扫掠圆孔网格与 RBF-FD 装配
│  ├─ frequency_system.py      # 约束算子与频域直接求解
│  └─ modal_resonance.py       # 共享模态有理分支
├─ training/
│  ├─ train_full_order_net.py  # 全阶背景训练
│  ├─ residual_refinement.py   # POD 增广训练及 FOM/POD 在线矫正
│  └─ physics_residual.py      # 可复用离散物理残差
├─ examples/cases.py           # flat / hollow / stiffened 统一配置
└─ visualization/              # 期刊绘图样式
```

---

## 参数化 many-query 与置信边界

固定结构流程之外，工程现在提供独立入口 `run_parametric_many_query.py`。该入口将网络输入扩展为 `(frequency, mu)`，其中 `mu=(young_scale, density_scale, pressure_scale, loss_factor)`，并为 `flat / hollow / stiffened` 三个结构执行：

- 未见参数测试；
- POD背景与共享模态有理分支；
- deep ensemble + split-conformal整条FRF置信带；
- 独立RBF-FD高保真统计对照；
- 离线成本、在线成本和盈亏平衡点统计。

详细公式、参数假设和运行命令见 [`PARAMETRIC_MANY_QUERY.md`](PARAMETRIC_MANY_QUERY.md)。原有固定结构入口及默认行为未改变。


---

## 论文结果部分：椭球 UQ 与统一绘图入口

论文方法论现已采用 **non-probabilistic ellipsoidal convex set**。为避免旧版
box-LHS / quantile 后处理与正文不一致，正式结果建议按以下顺序运行：

```powershell
# 1. 正式训练（可继续使用 enclosing box 提供训练覆盖）
python .\run_parametric_many_query.py --case all --representation pod --modal-modes 8 `
  --pod-modes 8 --n-labeled 48 --n-physics 160 --sweep-points 801 `
  --ensemble-size 5 --epochs 5000 --device cuda --out .\results\parametric_many_query

# 2. 单参数高保真物理解释
python .\generate_parameter_effect_study.py --results-root .\results\parametric_many_query `
  --case all --sweep-points 801 --out .\results\parameter_effects

# 3. 正式椭球 UQ、HF 统计、未见参数测试与 curve-wise calibration
python .\generate_ellipsoidal_uq_study.py --results-root .\results\parametric_many_query `
  --case all --n-uq 1000 --n-hf-statistics 50 --n-calibration 50 --n-test 50 `
  --sweep-points 801 --out .\results\ellipsoidal_uq

# 4. 四模型消融
python .\run_parametric_ablation.py --case all --physics-weight 0.02 `
  --out .\results\parametric_ablation --extra --representation pod --epochs 5000 `
  --ensemble-size 5 --sweep-points 801 --device cuda

# 5. 统一生成 Fig. 5--15
python .\generate_results_section_figures.py `
  --parametric-root .\results\parametric_many_query `
  --ellipsoid-root .\results\ellipsoidal_uq `
  --parameter-effect-root .\results\parameter_effects `
  --ablation-root .\results\parametric_ablation `
  --verification-csv .\results\verification\rbf_fd_verification.csv `
  --out .\results\manuscript_result_figures
```

完整的 Results 写作顺序、每张图对应的科学问题、指标和正文句式见
[`RESULTS_SECTION_WRITING_GUIDE.md`](RESULTS_SECTION_WRITING_GUIDE.md)。

新增的 `rbf_fd_full_order/uncertainty/ellipsoidal.py` 实现

\[
\boldsymbol\mu=\boldsymbol\mu^c+\mathbf D_\mu\mathbf L_\mu\boldsymbol\xi,
\qquad \|\boldsymbol\xi\|_2\le 1,
\]

并允许通过 `--e-rho-correlation` 与 `--correlation-values` 控制
\(\mathbf R_\mu\) 中 Young's-modulus/density 的依赖方向。最终论文中的
FRF bound 使用椭球样本上的 **minimum / maximum sample-estimated envelope**，
不再将 box quantile interval 称为非概率椭球响应边界。
