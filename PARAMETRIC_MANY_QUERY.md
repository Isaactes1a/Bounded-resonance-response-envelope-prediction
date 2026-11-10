# 参数化 many-query 神经算子与频响置信边界

## 1. 参数化范围

新增接口不改变原有固定结构训练入口。参数化入口采用固定几何、固定节点编号和固定边界条件下的四维参数

\[
\boldsymbol\mu=(s_E,s_\rho,s_p,\eta),
\]

并定义

\[
\mathbf K(\boldsymbol\mu)=s_E\mathbf K_0,
\qquad
\mathbf M(\boldsymbol\mu)=s_\rho\mathbf M_0,
\qquad
\mathbf f(\boldsymbol\mu)=s_p\mathbf f_0.
\]

其中 `young_scale`、`density_scale`、`pressure_scale` 和 `loss_factor` 均进入网络输入。默认区间为：

| 参数 | 默认范围 |
|---|---:|
| `young_scale` | 0.85--1.15 |
| `density_scale` | 0.90--1.10 |
| `pressure_scale` | 0.80--1.20 |
| `loss_factor` | 0.005--0.020 |

这一参数族对三个算例均成立，并且在线阶段不需要重新生成节点、RBF-FD权重或重新装配矩阵。该接口目前不把厚度和孔径作为随机参数，因为它们会改变刚度与质量矩阵的非统一缩放关系；若直接加入而不重新构造参数相关算子，会破坏高保真一致性。

## 2. 网络输入与增广表示

网络映射由原来的

\[
f\longmapsto \mathbf u(f)
\]

改为

\[
(f,\boldsymbol\mu)\longmapsto \mathbf u(f,\boldsymbol\mu).
\]

POD默认采用

\[
\mathbf u(\omega,\boldsymbol\mu)
=
\boldsymbol\Phi_{\mathrm{POD}}
\mathbf a_\theta(\omega,\boldsymbol\mu)
+
\mathbf V_{\mathrm{modal}}
\mathbf r_{\mathrm{rational}}(\omega,\boldsymbol\mu).
\]

参数化有理分支与全阶路线共享同一组名义模态。由于当前参数族满足统一刚度和质量缩放，模态振型保持不变，参数相关模态响应可精确写为

\[
\mathbf u_{\mathrm{modal}}
=
\sum_{r=1}^{N_m}
\boldsymbol\phi_r
\frac{s_p p_r}
{(1+\mathrm i\eta)s_E\lambda_r-s_\rho\omega^2}.
\]

因此共振频率随参数按

\[
f_r(\boldsymbol\mu)
=
\sqrt{\frac{s_E}{s_\rho}}f_{r,0}
\]

移动，而无需对每一个在线参数点重新求解特征值问题。POD网络只学习去除上述有理分支后的平滑背景响应。

## 3. 未见参数划分

`run_parametric_many_query.py` 使用互不重叠的Latin hypercube样本：

- `train`：生成监督数据和物理残差训练点；
- `calibration_unseen`：仅用于split-conformal校准；
- `test_unseen`：仅用于报告未见参数全场误差和曲线覆盖率；
- `uq_online`：用于快速生成参数传播边界；
- `hf_statistics`：独立高保真统计对照。

所有参数样本写入 `parameter_samples.csv`，可明确证明测试参数未进入训练。

## 4. 置信区间

每个算例训练若干个独立初始化的网络组成deep ensemble。对第 \(i\) 条校准频响曲线，采用同时覆盖整条曲线的非一致性分数

\[
S_i
=
\max_f
\frac{|y_i(f)-\bar y_i(f)|}
{\max[\sigma_i(f),s_0]}.
\]

由split-conformal有限样本分位数得到 \(q_{1-\alpha}\)，单条新曲线的置信带为

\[
\bar y(f)
\pm
q_{1-\alpha}\max[\sigma(f),s_0].
\]

最终频响边界同时考虑参数传播和代理模型不确定性：先对每个参数样本构造conformal曲线带，再沿参数样本取上下分位数。输出包括：

- 纯参数传播区间；
- conformal校准后的保守置信边界；
- ensemble spread；
- 基于区间相对宽度的显示型 `confidence_score`；
- 校准覆盖率和独立测试覆盖率。

`confidence_score` 只用于可视化区分区间宽窄，统计置信水平由split-conformal覆盖率定义。

## 5. 高保真统计对照

对独立的 `hf_statistics` 参数样本逐点调用约束RBF-FD直接求解器，并比较：

- 响应均值曲线；
- 标准差曲线；
- 下边界和上边界；
- 均值曲线峰值频率；
- 未见参数全场相对误差。

结果写入 `high_fidelity_statistical_comparison.json` 和 `frf_response_boundary.csv`。

## 6. 离线成本与盈亏平衡点

程序分别计时：

- RBF-FD装配；
- 模态分支准备；
- 高保真训练数据生成；
- ensemble训练；
- conformal校准数据生成；
- 单参数高保真完整扫频；
- 单参数神经算子ensemble完整扫频。

盈亏平衡查询次数定义为

\[
N_{\mathrm{BE}}
=
\left\lceil
\frac{C_{\mathrm{offline}}}
{C_{\mathrm{HF/query}}-C_{\mathrm{operator/query}}}
\right\rceil.
\]

结果写入 `cost_and_break_even.json`。当算子单次在线成本不低于直接求解时，程序将盈亏平衡点写为 `null`，不会给出虚假的加速结论。

## 7. 推荐运行

```bash
python run_parametric_many_query.py \
  --case all \
  --representation pod \
  --modal-modes 8 \
  --pod-modes 8 \
  --n-train-parameters 20 \
  --n-calibration-parameters 20 \
  --n-test-parameters 20 \
  --n-uq-parameters 1000 \
  --n-hf-statistics-parameters 50 \
  --n-labeled 24 \
  --n-physics 80 \
  --square-ndiv 30 \
  --plate-stencil 20 \
  --plate-poly-degree 2 \
  --beam-stencil 7 \
  --hollow-sweep-points 15 \
  --hollow-tangent-points-per-patch 16 \
  --hollow-stencil 20 \
  --hollow-poly-degree 2 \
  --sweep-points 801 \
  --ensemble-size 5 \
  --epochs 5000 \
  --device cuda
```

正式论文结果建议至少使用20个conformal校准参数点。样本过少时，95%置信水平的有限样本分位数非常粗糙，覆盖率结果只能作为流程检查。

## 8. 输出文件

每个结构目录包含：

- `parametric_operator.pt`：ensemble网络、POD基底和归一化信息；
- `parameter_samples.csv`：训练、校准、测试、在线UQ和高保真统计参数；
- `unseen_parameter_field_errors.csv`：未见参数全场误差；
- `frf_response_boundary.csv`：频响均值、参数区间、conformal区间和高保真边界；
- `high_fidelity_statistical_comparison.json`：统计曲线误差和覆盖率；
- `cost_and_break_even.json`：离线/在线成本和盈亏平衡点；
- `parametric_frf_boundary.png/.pdf`：红、黑、蓝、灰四色结果图；
- `metadata.json`：参数化假设和实验设置。
