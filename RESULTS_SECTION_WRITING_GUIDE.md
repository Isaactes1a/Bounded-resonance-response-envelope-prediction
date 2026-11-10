# 结果部分写作与作图指南
## Reusable RBF-FD + resonance-aware parametric surrogate + non-probabilistic ellipsoidal UQ

> **用途**：本文件直接服务于论文 `Numerical results and discussion` 的撰写与结果生成。  
> **核心原则**：结果部分按“证据链”组织，而不是按代码模块或三个算例逐个流水叙述。  
> **当前方法链**：
>
> \[
> \text{verified RBF-FD}
> \rightarrow
> \text{unseen-parameter surrogate}
> \rightarrow
> \text{pole/physics mechanism}
> \rightarrow
> \text{ellipsoidal FRF propagation}
> \rightarrow
> \text{resonance ranges}
> \rightarrow
> \text{surrogate calibration}
> \rightarrow
> \text{break-even}
> \]

---

# 1. 结果部分总结构

推荐正文目录：

```text
6. Numerical results and discussion
6.1 Computational models and RBF-FD verification
6.2 Physical effects of uncertain parameters on frequency responses
6.3 Surrogate accuracy for unseen parameter realizations
6.4 Contribution of the modal-rational branch and RBF-FD physics
6.5 Non-probabilistic ellipsoidal frequency-response envelopes
6.6 Envelope accuracy and resonance uncertainty ranges
6.7 Effect of parameter dependence and UQ sample convergence
6.8 Surrogate-error calibration
6.9 Computational efficiency and break-even analysis
6.10 Scope of the present evidence
```

建议结果正文约 **5000–7000 words**。核心不是平均分配篇幅，而是把最多篇幅留给 6.5–6.7，因为这部分才体现本文从一般参数化 FRF surrogate 进一步发展到非概率频域 UQ 的价值。

---

# 2. 每一节都使用同一种“结果段落结构”

每个结果 subsection 尽量按照以下五步写，不要只做“Figure x shows...”式图像复述：

1. **Question**：本节要验证哪一个方法论主张？
2. **Setup**：使用哪个结构、哪些参数、什么基准、多少样本？
3. **Observation**：图中首先看到什么规律？
4. **Quantification**：用 2–4 个数字量化，而不是只说 “good agreement”
5. **Interpretation + boundary**：为什么出现该现象？它在哪些区域仍然困难？

英文段落模板：

> This subsection examines whether [method component] preserves [specific quantity] under parameter variations that are excluded from model development. All methods are evaluated using the same [frequency grid / parameter realizations / high-fidelity reference]. Figure X shows that [principal observation]. Quantitatively, [metric 1] is [value], while [metric 2] remains below [value]. The remaining discrepancy is concentrated near [resonance / difficult case], where [mechanical explanation]. These results support [limited claim], while [explicit limitation] remains.

---

# 3. 6.1 Computational models and RBF-FD verification

## 3.1 本节回答的问题

> 后续 surrogate 和 UQ 使用的 RBF-FD direct solution 是否足够可信？

本节必须在任何神经网络结果之前完成。不能用 network physics residual 证明 RBF-FD 离散本身正确。

## 3.2 Figure 5 — 三结构与离散

脚本：

```bash
python generate_results_section_figures.py \
  --parametric-root results/parametric_many_query \
  --out results/manuscript_result_figures
```

输出：

```text
Fig05_structural_configurations.png/.pdf
```

图中只保留：

- solid square plate
- perforated square plate
- stiffened square plate
- 实际节点
- 孔边界 / stiffener line
- 必要几何信息

**不要重新画方法部分已经出现的 local stencil。**

## 3.3 Figure 6 — 独立验证

最终论文必须准备独立 analytical/FEM reference。统一输入表：

```text
results/verification/rbf_fd_verification.csv
```

必须至少包含：

```csv
case,node_count,mode,rbf_frequency_hz,reference_frequency_hz,mac
flat,441,1,...,...,...
flat,961,1,...,...,...
...
```

如果没有独立 FEM/解析结果，绘图脚本会跳过 Figure 6，而不会把 surrogate 或 RBF-FD 自身的细网格解伪装成“独立验证”。

建议 Figure 6：

- (a) natural-frequency convergence vs node count
- (b) finest-grid modal-frequency error vs independent reference

主表建议：

| Mode | Independent reference | RBF-FD | error | MAC |
|---|---:|---:|---:|---:|

### 本节写作重点

第一段：说明网格、stencil、频域网格和 independent reference。  
第二段：报告前 4–6 阶固有频率误差和 MAC。  
第三段：报告 deterministic FRF，尤其是 resonance zoom。  
最后一句应限定为：

> These results establish the RBF-FD model as a sufficiently converged high-fidelity reference for the subsequent learning and uncertainty studies.

不要写“RBF-FD universally outperforms FEM”。

---

# 4. 6.2 Physical effects of uncertain parameters on FRFs

## 4.1 为什么要单独做这一节

当前不确定向量为

\[
\boldsymbol\mu=[s_E\ s_\rho\ s_p\ \eta]^{\mathsf T}
\]

四个参数在物理上承担不同作用：

- \(s_E\)：主要移动 resonance դեպի高频
- \(s_\rho\)：主要移动 resonance դեպի低频
- \(s_p\)：主要改变幅值
- \(\eta\)：主要抑制峰值并改变峰宽

这一节用于证明 modal-rational branch 的设计来自真实结构动力学规律，而不是为了提高网络指标而人为加入。

## 4.2 生成数据

先完成正式 parametric checkpoint，然后运行：

```bash
python generate_parameter_effect_study.py \
  --results-root results/parametric_many_query \
  --case all \
  --sweep-points 801 \
  --out results/parameter_effects
```

该脚本使用 **direct constrained RBF-FD**，不是 surrogate。

输出：

```text
results/parameter_effects/<case>/
├─ parameter_effect_frf.csv
├─ parameter_effect_frf.png/.pdf
└─ parameter_effect_summary.json
```

## 4.3 Figure 7

统一主图建议选择最复杂的 `stiffened` case：

```bash
python generate_results_section_figures.py \
  --representative-case stiffened
```

输出：

```text
Fig07_single_parameter_physical_effects.png/.pdf
```

四个 panel：

- (a) \(s_E\)
- (b) \(s_\rho\)
- (c) \(s_p\)
- (d) \(\eta\)

### 写作重点

不要逐条描述每一条线。要概括为两个物理机制：

**horizontal resonance drift**

\[
f_r \propto \sqrt{s_E/s_\rho}
\]

以及 **vertical amplification control** 主要由 \(s_p\) 和 \(\eta\) 控制。

结尾自然引向 6.3：

> This separation between pole-location and amplitude effects motivates the subsequent decomposition of the response into a parameter-dependent rational modal contribution and a smoother learned background.

---

# 5. 6.3 Surrogate accuracy for unseen parameter realizations

## 5.1 数据必须满足的条件

- test parameter realizations 整条曲线均未用于训练
- test 与 calibration 分开
- test 参数来自论文正式采用的 ellipsoid
- 使用 complete FRF，而不是随机拆 frequency points

正式椭球结果通过：

```bash
python generate_ellipsoidal_uq_study.py \
  --results-root results/parametric_many_query \
  --case all \
  --n-uq 1000 \
  --n-hf-statistics 50 \
  --n-calibration 50 \
  --n-test 50 \
  --sweep-points 801 \
  --out results/ellipsoidal_uq
```

## 5.2 Figure 8 — 完整 unseen error distribution

输出：

```text
Fig08_unseen_parameter_error_distributions.png/.pdf
```

四个指标：

1. observation FRF relative \(L_2\)
2. full reduced-response relative \(L_2\)
3. peak-frequency error
4. peak-amplitude error

正文必须至少报告：

- median
- 95th percentile
- worst case 或 max
- 三结构差异

推荐 Table：

| Case | median FRF | P95 FRF | median full-field | peak-f MAE | peak-amplitude error |
|---|---:|---:|---:|---:|---:|

## 5.3 Figure 9 — 代表性完整 FRF + resonance zoom

输出：

```text
Fig09_representative_unseen_frfs.png/.pdf
```

每个结构一行：

| left | right |
|---|---|
| complete FRF | dominant resonance zoom |

颜色：

- black：RBF-FD HF
- red：proposed surrogate

正文不要只讨论“曲线重合”。必须说明：

- off-resonance background 是否准确
- peak location 是否发生 artificial horizontal shift
- narrow resonance 是否出现 smoothing
- 哪个结构最困难

---

# 6. 6.4 Contribution of pole branch and RBF-FD physics

## 6.1 四个受控模型

必须保持相同：

- parameter samples
- random seed
- frequency grid
- architecture width/depth
- POD modes
- epochs

只改变两件事：

| Model | pole branch | physics residual |
|---|---:|---:|
| Direct-data | no | no |
| Direct-physics | no | yes |
| Pole-data | yes | no |
| Full | yes | yes |

运行：

```bash
python run_parametric_ablation.py \
  --case all \
  --physics-weight 0.02 \
  --out results/parametric_ablation \
  --extra \
  --representation pod \
  --epochs 5000 \
  --ensemble-size 5 \
  --n-train-parameters 20 \
  --n-calibration-parameters 20 \
  --n-test-parameters 20 \
  --n-uq-parameters 500 \
  --n-hf-statistics-parameters 30 \
  --sweep-points 801 \
  --device cuda
```

先检查命令而不执行：

```bash
python run_parametric_ablation.py ... --dry-run
```

## 6.2 Figure 10

输出：

```text
Fig10_four_model_ablation.png/.pdf
```

当前 plotter 使用四项现有输出：

- mean unseen full-field error
- mean peak-frequency error
- upper-boundary relative error
- curve-wise coverage

### 结果解释的预期逻辑

不要简单写：

> Full model is best.

需要分别回答：

**Pole branch 做了什么？**

重点比较：

- Direct-data vs Pole-data
- Direct-physics vs Full

如果 peak-frequency error 明显下降，说明 pole branch 主要解决 moving resonance representation。

**RBF-FD physics 做了什么？**

重点比较：

- Direct-data vs Direct-physics
- Pole-data vs Full

如果 upper error / unseen error / residual consistency 改善，说明 physics term 的主要作用是 unseen consistency，而不是简单压低 training error。

允许出现 negative result：

> The physics term does not reduce every aggregate metric, but it consistently suppresses the upper error tail / residual inconsistency.

---

# 7. 6.5 Non-probabilistic ellipsoidal FRF envelopes

## 7.1 重要：最终论文不要继续使用旧 box-quantile 图

旧入口：

```text
run_parametric_many_query.py
```

中的 `frf_response_boundary.csv` 仍属于历史/训练阶段的 **box sampling + quantile interval**。

最终论文正式的 non-probabilistic ellipsoidal result 应使用：

```text
generate_ellipsoidal_uq_study.py
```

输出：

```text
ellipsoidal_envelope.csv
ellipsoidal_uq_curves.npz
```

这里定义的是：

- parameter domain：ellipsoid
- boundary estimator：sample minimum / maximum
- 不称为 percentile CI
- 不称为 certified global bound

## 7.2 Figure 11 — 三结构 FRF envelope

输出：

```text
Fig11_ellipsoidal_frf_envelopes.png/.pdf
```

图层：

- gray：individual ellipsoidal surrogate FRFs
- black dashed：independent HF sample bounds
- red：surrogate ellipsoidal sample-estimated bounds

### 本节正文结构

第一段：说明 \(\mathcal U_E\)、\(N_{\rm UQ}\)、boundary fraction 和 frequency grid。  
第二段：比较整体 envelope。  
第三段：集中讨论 resonance 区域。  
第四段：强调“不同 frequency 的 upper bound 可能由不同 parameter realization 控制”。

必须使用：

> sample-estimated envelope

而不是：

> exact global envelope

除非以后真正做 constrained optimization。

---

# 8. 6.6 Envelope accuracy and resonance uncertainty ranges

## 8.1 Envelope metrics

从：

```text
ellipsoidal_uq_summary.json
ellipsoidal_envelope.csv
```

报告：

- lower-boundary relative \(L_2\)
- upper-boundary relative \(L_2\)
- width relative \(L_2\)
- frequency-resolved maximum discrepancy

建议 Table：

| Case | \(e_L\) | \(e_U\) | \(e_W\) | max boundary error |
|---|---:|---:|---:|---:|

## 8.2 Figure 12 — resonance frequency/amplitude ranges

输出：

```text
Fig12_resonance_frequency_amplitude_ranges.png/.pdf
```

对应当前方法中的：

\[
\mathcal F_r=[f_{r,L}\ f_{r,U}]
\]

\[
\mathcal A_r=[a_{r,L}\ a_{r,U}]
\]

左图：

- horizontal resonance uncertainty

右图：

- vertical peak-amplitude uncertainty

### 写作重点

这一节要明确解释为什么 average FRF error 不够：

> A narrow horizontal resonance shift can be mechanically important while contributing only weakly to a global curve norm.

因此 resonance range 是 vibration avoidance 的更直接结果。

---

# 9. 6.7 Effect of parameter dependence and UQ sample convergence

## 9.1 为什么必须有相关性实验

如果正文采用

\[
\mathbf R_\mu
\]

定义 correlated ellipsoid，但结果从不改变 \(\mathbf R_\mu\)，审稿人很容易认为 ellipsoid 只是另一种 random sampler。

因此至少需要一个 controlled E-density correlation study。

当前脚本默认：

```text
rho_E,rho = -0.5
rho_E,rho = 0
rho_E,rho = +0.5
```

通过：

```bash
python generate_ellipsoidal_uq_study.py \
  ... \
  --correlation-values -0.5 0 0.5
```

## 9.2 Figure 13(a) — ellipsoid orientation

输出 Figure 13 左侧：

```text
envelope width vs frequency
```

不同 \(\rho_{E,\rho}\) 对应不同线。

讨论重点：

\[
f_r\propto\sqrt{s_E/s_\rho}
\]

因此 E 与 density 的依赖方向会直接改变 reachable resonance-frequency range。

不要写：

> correlated ellipsoid is more accurate

除非存在真实 joint-sample evidence。

应写：

> incorporating dependence changes the admissible response domain and removes parameter combinations that are inconsistent with the prescribed convex geometry.

## 9.3 Figure 13(b) — UQ sample convergence

脚本自动生成：

```text
ellipsoidal_sample_convergence.csv
```

比较：

- lower boundary
- upper boundary
- width

相对于 full UQ set 的变化。

输出：

```text
Fig13_ellipsoid_dependence_and_sample_convergence.png/.pdf
```

正文需要说明：

> The input ellipsoid is defined analytically, whereas its propagated FRF extrema are numerically approximated from a finite realization set.

因此 sample convergence 与 surrogate error 是两种不同的误差来源。

---

# 10. 6.8 Surrogate-error calibration

## 10.1 parameter uncertainty 与 surrogate error 必须分开

正文中始终区分：

**parameter-induced envelope**

\[
\mathcal E_{\rm param}(\omega)
=
[y_L^E(\omega)\ y_U^E(\omega)]
\]

和

**surrogate-error band**

\[
\mathcal I_{\rm surr}(\omega,\boldsymbol\mu)
\]

不能把二者都叫 95% confidence interval。

## 10.2 校准样本数

正式结果建议：

```text
N_cal >= 40–60 complete parameter curves
```

而不是 40–60 frequency points。

运行：

```bash
python generate_ellipsoidal_uq_study.py \
  ... \
  --n-calibration 50 \
  --n-test 50 \
  --confidence 0.95
```

## 10.3 Figure 14

输出：

```text
Fig14_surrogate_error_calibration.png/.pdf
```

左图：

- HF curve：black
- ensemble mean：red
- \(\mathcal I_{\rm surr}\)：blue transparent band

右图：

- nominal coverage
- empirical curve-wise coverage
- diagonal reference

正文报告：

- raw/empirical calibration coverage
- independent test coverage
- conformal multiplier
- mean band width

不要把 pointwise coverage 当 curve-wise coverage。

---

# 11. 6.9 Computational efficiency and break-even

## 11.1 不允许只报告 inference speedup

必须同时报告：

\[
C_{\rm offline}
\]

包括：

- RBF-FD assembly
- modal preparation
- HF training data
- ensemble training
- calibration HF solves

以及：

- \(C_{\rm HF/query}\)
- \(C_{\rm surr/query}\)

最终：

\[
N_{\rm BE}
=
\left\lceil
\frac{C_{\rm offline}}
{C_{\rm HF/query}-C_{\rm surr/query}}
\right\rceil
\]

## 11.2 Figure 15

输出：

```text
Fig15_cumulative_cost_break_even.png/.pdf
```

横轴：

```text
number of complete parameter queries
```

纵轴：

```text
cumulative wall-clock cost
```

每个结构：

- solid line：direct HF
- dashed line：surrogate workflow

正文先承认 offline cost，再讨论 break-even：

> Although a single surrogate sweep is substantially cheaper than direct RBF-FD evaluation, the relevant quantity for a many-query workflow is the cumulative cost after including the one-time offline investment.

---

# 12. 6.10 Scope of the present evidence

建议 300–500 words。

必须主动写清：

1. 当前是 fixed topology 下的 in-domain parameter generalization
2. 三个 geometry 分开训练，共享的是方法而不是 weights
3. 当前 ellipsoidal envelope 是 finite-sample estimate
4. \((s_E,s_\rho)\) proportional scaling 保持 nominal modes；local/non-proportional uncertainty 需要 parameter-dependent basis
5. 若没有实验数据，不要称 experimental validation
6. credible-set inference from scarce measurements 目前是可兼容扩展，不是本文实际执行的输入域识别方法

---

# 13. 主图清单与代码对应

| Figure | 内容 | 主要数据源 | 生成方式 |
|---|---|---|---|
| Fig. 5 | 三结构和节点 | geometry builders | `generate_results_section_figures.py` |
| Fig. 6 | RBF-FD independent verification | external analytical/FEM CSV | same |
| Fig. 7 | 单参数 FRF 物理影响 | direct RBF-FD | `generate_parameter_effect_study.py` + same |
| Fig. 8 | unseen error distributions | ellipsoid test | `generate_ellipsoidal_uq_study.py` + same |
| Fig. 9 | unseen FRF + resonance zoom | ellipsoid test | same |
| Fig. 10 | four-model ablation | four controlled runs | `run_parametric_ablation.py` + same |
| Fig. 11 | ellipsoidal FRF envelopes | ellipsoid UQ + HF | ellipsoid script + same |
| Fig. 12 | resonance intervals | ellipsoid UQ | same |
| Fig. 13 | dependence + UQ convergence | ellipsoid UQ | same |
| Fig. 14 | calibration | ellipsoid calibration/test | same |
| Fig. 15 | cumulative cost / break-even | formal training timing | same |

统一最终绘图：

```bash
python generate_results_section_figures.py \
  --parametric-root results/parametric_many_query \
  --ellipsoid-root results/ellipsoidal_uq \
  --parameter-effect-root results/parameter_effects \
  --ablation-root results/parametric_ablation \
  --verification-csv results/verification/rbf_fd_verification.csv \
  --representative-case stiffened \
  --out results/manuscript_result_figures
```

如果某项昂贵实验尚未完成，脚本不会伪造结果，而是在：

```text
figure_manifest.csv
```

中标记 `skipped`。

---

# 14. 推荐正式计算顺序

```text
Step 1
RBF-FD independent verification
        ↓
Step 2
formal parametric operator training
        ↓
Step 3
single-parameter direct-HF physical study
        ↓
Step 4
ellipsoidal UQ + unseen test + calibration + HF statistics
        ↓
Step 5
controlled four-model ablation
        ↓
Step 6
all manuscript result figures
        ↓
Step 7
write Results from quantitative tables, not from visual impression
```

正式训练建议：

```bash
python run_parametric_many_query.py \
  --case all \
  --representation pod \
  --modal-modes 8 \
  --pod-modes 8 \
  --n-train-parameters 20 \
  --n-calibration-parameters 20 \
  --n-test-parameters 20 \
  --n-uq-parameters 500 \
  --n-hf-statistics-parameters 30 \
  --n-labeled 48 \
  --n-physics 160 \
  --sweep-points 801 \
  --ensemble-size 5 \
  --epochs 5000 \
  --device cuda \
  --out results/parametric_many_query
```

注意：这一步的 box samples 主要承担**训练覆盖**。最终论文中 ellipsoidal UQ envelope 由后续 `generate_ellipsoidal_uq_study.py` 生成。

---

# 15. 每节需要在正文中真正出现的数字

## 6.1

- first 4–6 eigenfrequency errors
- MAC
- deterministic FRF peak error

## 6.2

不要求大量误差数字，重点报告峰值移动趋势，例如：

- \(s_E\) increase → peak shift
- \(s_\rho\) increase → opposite shift
- \(s_p\) amplitude scaling
- \(\eta\) peak suppression

## 6.3

- median
- P95
- maximum/worst representative
- peak frequency error
- peak amplitude error

## 6.4

四模型相对改善百分比：

> Pole-data reduces peak-frequency error by X% relative to Direct-data.

> Adding RBF-FD physics changes P95 unseen error from X to Y.

## 6.5–6.6

- \(e_L\)
- \(e_U\)
- \(e_W\)
- max local boundary discrepancy
- \(\mathcal F_r\)
- \(\mathcal A_r\)

## 6.7

- correlation-induced change in resonance/envelope width
- UQ sample convergence threshold

## 6.8

- nominal coverage
- empirical test coverage
- conformal multiplier
- mean band width

## 6.9

- offline total
- direct query
- surrogate query
- marginal speedup
- \(N_{\rm BE}\)

---

# 16. 结果写作中避免的表达

避免：

> The curves are almost identical.

改成：

> The median FRF error is X%, while the 95th percentile remains below Y%; the largest discrepancies are localized around the second resonance.

避免：

> The uncertainty interval is accurate.

改成：

> The sample-estimated lower and upper envelopes reproduce the independent HF bounds with relative errors of X and Y, respectively.

避免：

> The 95% confidence interval describes structural uncertainty.

如果说的是 parameter-induced ellipsoid，改成：

> non-probabilistic parameter-induced response envelope

只有 conformal band 才讨论 coverage。

避免：

> The method is 1000 times faster.

改成：

> The marginal query cost is reduced by X, while the cumulative-cost curves cross after approximately \(N_{\rm BE}\) complete parameter sweeps.

---

# 17. 当前代码与论文方法的一致性检查

在最终出图之前必须检查：

- [ ] 训练网络输入确实包含 \(s_E,s_\rho,s_p,\eta\)
- [ ] final UQ 采用 `EllipsoidalConvexSet`
- [ ] final UQ bound 使用 sample min/max，不是 percentile
- [ ] correlation matrix \(\mathbf R_\mu\) 被记录
- [ ] calibration/test/UQ/HF samples 独立
- [ ] \(\mathcal E_{\rm param}\) 与 \(\mathcal I_{\rm surr}\) 分图或明确区分
- [ ] all test parameter curves unseen
- [ ] Fig. 6 reference 独立于 RBF-FD
- [ ] ablation 只改变 pole / physics 两个因素
- [ ] speedup 包括 offline cost
- [ ] figure fonts = Times New Roman 10 pt
- [ ] final colors = red / black / blue / gray
- [ ] resonance plots 使用相同 frequency zoom convention

---

# 18. 最终 Results 的主叙事

整章始终围绕下面这句话推进：

> The numerical evidence is organized to establish, in sequence, the fidelity of the reusable RBF-FD discretization, the unseen-parameter accuracy and resonance resolution of the response surrogate, and the reliability and computational value of its non-probabilistic ellipsoidal uncertainty propagation.

随后所有图表分别回答：

```text
Is the high-fidelity model trustworthy?
↓
Does the surrogate generalize to unseen μ?
↓
Why do the pole branch and discrete physics matter?
↓
Does the surrogate preserve the entire ellipsoidal FRF envelope?
↓
Does it preserve resonance-frequency and amplitude ranges?
↓
Are finite-sample envelope and surrogate-error uncertainties separately controlled?
↓
When does the offline investment become worthwhile?
```

只要结果部分始终围绕这条证据链组织，整篇文章的方法论、结果和结论就能形成闭环。
