import numpy as np

path = (
    "/Users/netanelblumenfeld/Downloads/11G/data_feat/p0_1/FingerRub_1s_wl32_doppl.npy"
)
path1 = (
    "/Users/netanelblumenfeld/Downloads/11G/data_feat/p1/FingerRub_1s_wl32_doppl.npy"
)
data = np.load(path)
data1 = np.load(path1)
print(data.shape)
print(data1.shape)
