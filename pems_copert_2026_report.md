# COPERT 2026 CO₂计算

依据《EMEP/EEA air pollutant emission inventory guidebook 2023 – Update 2026》PDF表3-17，选择 Passenger cars / Petrol Medium / Euro 6 a/b/c，CO₂因子为 **182.309 g/km**。逐秒预测值为 `182.309 × v(km/h) / 3600`。

同一清洗数据和同一1463条测试集：R²=-1.3900，RMSE=1.9866 g/s，MAE=1.5837 g/s。

按图片误差率公式，在速度>5 km/h且实测排放因子>0的1271条样本上，平均误差率=2.9760（297.60%），中位数=0.9546（95.46%）。
