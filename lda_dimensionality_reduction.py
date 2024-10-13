# lda_dimensionality_reduction.py
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# 生成隨機數據
data = np.random.rand(100, 5)
labels = np.random.randint(0, 2, 100)

# 應用 LDA 降維
lda = LinearDiscriminantAnalysis(n_components=1)
reduced_data = lda.fit_transform(data, labels)

# 可視化結果
plt.scatter(reduced_data, np.zeros_like(reduced_data), c=labels, cmap='viridis')
plt.show()