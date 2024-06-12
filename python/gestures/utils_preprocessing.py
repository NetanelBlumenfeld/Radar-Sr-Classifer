import torch
import torch.nn.functional as F


class ComplexToReal(torch.nn.Module):
    def __init__(self):
        super(ComplexToReal, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack((x.real, x.imag), dim=3)


class DownSampleOffLine(torch.nn.Module):
    def __init__(self, D: int, original_dim: bool = False):
        super(DownSampleOffLine, self).__init__()
        self.D = D
        self.original_dim = original_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 6  # (N, 5, 2, 2,32, 492)
        if self.D == 1:
            return x
        res_ds = x[:, :, :, :, :: self.D, :: self.D]
        if self.original_dim:
            d0, d1, d2, d3, d4, d5 = res_ds.size()
            res_up = torch.empty(
                d0,
                d1,
                d2,
                d3,
                d4 * self.D,
                d5 * self.D,
                dtype=res_ds.dtype,
                device=res_ds.device,
            )

            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    res_up[i, j] = F.interpolate(
                        res_ds[i, j],
                        scale_factor=(self.D, self.D),
                        mode="bicubic",
                    )
            return res_up
        return res_ds


class DopplerMaps(torch.nn.Module):
    def __init__(self):
        super(DopplerMaps, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 6
        x_real = x[:, :, :, 0, :, :]
        x_image = x[:, :, :, 1, :, :]
        doppler_maps = torch.abs(
            torch.fft.fftshift(
                torch.fft.fft(torch.complex(x_real, x_image), dim=-2), dim=-2
            )
        )
        return doppler_maps


class Normalize(torch.nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    min_val = x[i, j, k].min()
                    max_val = x[i, j, k].max()
                    x[i, j, k] = (x[i, j, k] - min_val) / (max_val - min_val + 1e-8)
        # print(x.dtype)
        return x


class ToTensor(torch.nn.Module):
    def __init__(self, dtype=torch.float):
        super(ToTensor, self).__init__()
        self.dtype = dtype

    def forward(self, x):
        # Convert x to a tensor if it's not already one
        return x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.cfloat)
