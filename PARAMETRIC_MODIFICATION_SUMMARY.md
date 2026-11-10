# 参数化模块修改摘要

1. 新增固定几何参数族 `mu=(young_scale, density_scale, pressure_scale, loss_factor)`，并使 `mu` 与频率共同进入全阶/POD网络。
2. 新增精确缩放高保真接口：`K(mu)=s_E K0`、`M(mu)=s_rho M0`、`f(mu)=s_p f0`。
3. 新增参数化模态有理层，共振频率按 `sqrt(s_E/s_rho)`移动，在线阶段无需重新装配或重新特征值求解。
4. POD路线保持“POD背景 + 与全阶共享的模态有理分支”。
5. 新增独立训练、校准、测试、在线UQ和高保真统计样本划分。
6. 新增deep ensemble与split-conformal整条FRF同时置信带。
7. 新增三个结构的参数传播边界、高保真统计对照、覆盖率和置信度诊断。
8. 新增离线成本、单查询成本和盈亏平衡查询次数报告。
9. 原有 `run_frequency_validation.py` 固定结构流程保持不变；新入口为 `run_parametric_many_query.py`。
10. 全部15项回归测试通过，并完成三个结构的小规模端到端流程检查。
