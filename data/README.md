# 数据说明

项目使用 `sklearn.datasets.load_wine` 随 scikit-learn 离线加载的数据，不在仓库重复保存原始 CSV。

- 样本单位：一条意大利葡萄酒化学分析记录
- 样本数：178
- 特征数：13 个连续数值特征
- 标签：cultivar_0、cultivar_1、cultivar_2 三个品种类别
- 来源：UCI Wine Data Set
- 许可：CC BY 4.0
- 划分：固定随机种子 42，分层保留 20% 测试集；剩余训练集用于 5 折分层交叉验证

数据集不含用户、时间或重复主体标识，因此没有同一主体跨集合的已知风险。所有 StandardScaler 参数仅在每个训练折内拟合。

