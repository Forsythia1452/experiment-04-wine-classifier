# 测试记录

测试日期：2026-09-30  
环境：Windows、Python 3.12.14、scikit-learn 1.7.2、Streamlit 1.50.0。

执行命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

结果：`27 passed`，0 项失败，0 项警告。

## 覆盖范围

| 类型 | 数量 | 证据 |
|---|---:|---|
| 数据与划分 | 6 | 178×13、三类、无缺失、固定种子、集合不交叉、分层比例 |
| 模型评价 | 4 | 两种 Pipeline、同协议比较、最终指标、模型哈希 |
| Pipeline 与 Schema | 7 | 原始输入、缺失、未知、非数字、无穷大、列顺序、加载一致性 |
| 应用 | 6 | 无异常、摘要指标、标签页、13 项表单、预测、证据表 |
| 自主功能 | 4 | 正常、中风险、高风险、预测结果携带越界证据 |

## 最终评价证据

逻辑回归以 CV macro F1 0.9796 入选。36 条保留测试样本只评价一次，Accuracy 0.9722，macro Precision 0.9778，macro Recall 0.9667，macro F1 0.9710。混淆矩阵为 `[[12,0,0],[0,14,0],[0,1,9]]`。

模型文件 `wine_pipeline.joblib` 的 SHA256 与 `model_meta.json` 一致。自动化测试还确认相同 fixture 在加载前后的预测一致。

