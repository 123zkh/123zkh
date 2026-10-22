# COPERT Tier 3能源消耗函数与PEMS校准

参数来自2026年Appendix 4：Passenger Cars / Petrol / Medium / Euro 6 a/b/c，GDI与PFI的EC参数相同。EC函数按PDF公式(25)计算，并用汽油热值43.774 MJ/kg、汽油CO₂因子3.168 kg/kg换算。

PEMS校准采用训练集正速度、正排放样本的乘法系数，校准系数 **k=0.685381**；测试集未参与校准。

未校准测试集：R²=0.0017，RMSE=1.2839 g/s，MAE=1.0147 g/s。

校准后测试集：R²=0.3670，RMSE=1.0224 g/s，MAE=0.7777 g/s。

校准后误差率（速度>5 km/h且实测因子>0）：平均=137.53%，中位数=38.22%，超过20%比例=72.46%。
