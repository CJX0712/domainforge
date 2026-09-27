# DomainForge

**模块化域自适应（Domain Adaptation / Transfer Learning）系统 · 作者：晨星**

CORAL / TCA / JDA / KLIEP 四类经典对齐算法纯 numpy 手写（零重型依赖、离线可跑），
旗舰方法 **SAFuse**（Safeguarded Accuracy-Filtered fusion）：逐方法 HPO + 增益门控软投票 +
非劣守护回退，可选 `adapt`（TensorFlow 后端）做跨实现对拍。

## 一键复现

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.lock.txt   # Windows
# 或: pip install -r requirements.lock.txt
python -m pytest -q -W ignore::UserWarning          # 测试
python -m domainforge.examples.run_demo             # 端到端基准 -> benchmark.json
python -m domainforge.cli benchmark --out bench.json
```

## 架构

```
cli → pipeline → { data, hpo, fusion, alignment, baselines, eval } → core
```

| 模块 | 职责 |
|------|------|
| `core/`      | types(DomainSplit/MethodResult/BenchmarkRow) · errors(E100~E500) · config(ENV_DOMAINFORGE_* 覆盖) · interfaces(Protocol) |
| `data/`      | 4 组合成域偏移数据集（协变量偏移/旋转噪声/尺度漂移/标签偏移）+ npz/csv 加载 |
| `alignment/` | CORAL（协方差白化-重着色）· TCA（边际 MMD 广义特征问题）· JDA（类条件 MMD 迭代伪标签）· KLIEP-lite（判别器密度比实例重加权） |
| `baselines/` | source_only 基线 · `adapt` 可选对拍后端（惰性探测，不可用即 skipped，不造假行） |
| `hpo/`       | Optuna TPE 逐方法超参搜索（目标 = 极小标注目标验证集 acc） |
| `fusion/`    | **SAFuse** 旗舰（见下） |
| `pipeline/`  | 跨方法基准、聚合、JSON 落盘、定宽表格 |

核心契约：目标测试标签 **私有**，任何方法只能看到源域 + 极小（默认 20 条）标注目标验证样本。

## SAFuse 旗舰（创新点）

1. **逐方法 HPO**：TPE 在极小验证集上为每个成员选参；
2. **增益门控软投票**：成员权重 g_m = max(0, val_acc(m) − val_acc(source_only))，
   打不过 source-only 的成员权重为 **0**，坏成员无法拖垮集成；
3. **非劣守护**：集成在验证集上若劣于最强单成员（容差 0.01）→ 回退最强单成员。
   回退是一等公民结果，不是失败。

## 基准

固定 seed=42、HPO_TRIALS=16，结果见 `benchmark.json`（域 × 方法 30 行 + 聚合）。
数值全部可由 `python -m domainforge.examples.run_demo` 一条命令复现。

## 质量门

- pytest 单测全绿（含 CORAL 协方差对齐、TCA/JDA 广义特征问题残差、KLIEP 权重分布、
  旗舰守护回退、管线确定性、错误码、CLI 冒烟）
- `ruff format --check` / `ruff check` 通过
- 可选后端不可用时基准照常产出（skipped 行，诚实标注）

## 作者

晨星（Morningstar）· 2026
