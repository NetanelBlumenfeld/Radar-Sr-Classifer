import torch
import torch.nn.functional as F


class ToTensor(torch.nn.Module):
    def __init__(self):
        super(ToTensor, self).__init__()

    def forward(self, x) -> torch.Tensor:
        # Convert x to a tensor if it's not already one
        x = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.cfloat)
        x = x.permute(0, 3, 1, 2)  # (5,32,492,2) -> (5,2,32,492)

        return x


class NormalizeOneSample1(torch.nn.Module):
    def __init__(self):
        super(NormalizeOneSample1, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        # x dims (5,2,32,492)
        for i in range(x.shape[0]):
            for k in range(x.shape[1]):
                img = x[i, k, :, :]
                img_abs = torch.abs(img)
                min_val = img_abs.min()
                max_val = img_abs.max()
                x[i, k, :, :] = (img - min_val) / (max_val - min_val + 1e-8)
        return x


class NormalizeBatch1(torch.nn.Module):
    def __init__(self):
        super(NormalizeBatch1, self).__init__()
        self.one_sample_norm = NormalizeOneSample()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO removing the for loop
        assert x.ndim == 5
        # x dims (N,5,2,32,492)
        for i in range(x.shape[0]):
            x[i] = self.one_sample_norm(x[i])
        return x


class NormalizeOneSample(torch.nn.Module):
    def __init__(self):
        super(NormalizeOneSample, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        x_abs = torch.abs(x)
        min_val = (
            x_abs.view(x.shape[0], x.shape[1], -1)
            .min(dim=-1, keepdim=True)[0]
            .view(x.shape[0], x.shape[1], 1, 1)
        )
        max_val = (
            x_abs.view(x.shape[0], x.shape[1], -1)
            .max(dim=-1, keepdim=True)[0]
            .view(x.shape[0], x.shape[1], 1, 1)
        )
        x = (x - min_val) / (max_val - min_val + 1e-8)
        return x


class NormalizeBatch(torch.nn.Module):
    def __init__(self):
        super(NormalizeBatch, self).__init__()
        self.one_sample_norm = NormalizeOneSample()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5
        x = torch.stack([self.one_sample_norm(sample) for sample in x])
        return x


class ComplexToRealOneSample(torch.nn.Module):
    def __init__(self):
        super(ComplexToRealOneSample, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        # x dims (5,2,32,492)
        # x dims output (5,2,2,32,492) (sequence_length, sensors,channels, H, W)
        return torch.stack((x.real, x.imag), dim=2)


class ComplexToRealBatch(torch.nn.Module):
    def __init__(self):
        super(ComplexToRealBatch, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5
        # x dims (N,5,2,32,492)
        # x dims output (N,5,2,2,32,492) (batch sequence_length, sensors,channels, H, W)
        return torch.stack((x.real, x.imag), dim=3)


class RealToComplexOneSample(torch.nn.Module):
    def __init__(self):
        super(RealToComplexOneSample, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5
        # x dims (2,5,32,492,2)
        # x dims output (5,32,492,2)
        x_real = x[0]
        x_imag = x[1]
        return torch.complex(x_real, x_imag)


class RealToComplexBatch(torch.nn.Module):
    def __init__(self):
        super(RealToComplexBatch, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 6
        # x dims (N,5,2,2,32,492)
        # x dims output (N,5,2,32,492)
        x_real = x[:, :, :, 0]
        x_imag = x[:, :, :, 1]
        return torch.complex(x_real, x_imag)


class DopplerMapOneSample(torch.nn.Module):
    def __init__(self):
        super(DopplerMapOneSample, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        # x dims (5,2,32,492)
        doppler_maps = torch.abs(torch.fft.fftshift(torch.fft.fft(x, dim=2), dim=2))
        return doppler_maps


class DopplerMapBatch(torch.nn.Module):
    def __init__(self):
        super(DopplerMapBatch, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5
        # x dims (N,5,2,32,492)
        doppler_maps = torch.abs(torch.fft.fftshift(torch.fft.fft(x, dim=3), dim=3))
        return doppler_maps


class DownSampleOneSample(torch.nn.Module):
    def __init__(self, dx: int, dy: int, original_dims: bool = False):
        super(DownSampleOneSample, self).__init__()
        self.dx = dx
        self.dy = dy
        self.original_dims = original_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4  # (5,2,32,492)
        if self.dx == 1 and self.dy == 1:
            return x
        ds_data = x[:, :, :: self.dy, :: self.dx]
        if self.original_dims:
            return self._interpolate(ds_data)
        return ds_data

    def _interpolate(self, input_tensor):
        # Separate real and imaginary parts
        real_part = input_tensor.real
        imaginary_part = input_tensor.imag

        # Calculate the new size based on scale factors
        new_height = int(real_part.shape[-2] * self.dy)
        new_width = int(real_part.shape[-1] * self.dx)

        # Perform interpolation on both real and imaginary parts
        interpolated_real = F.interpolate(
            real_part,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        interpolated_imaginary = F.interpolate(
            imaginary_part,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

        # Combine interpolated real and imaginary parts back into a complex tensor
        interpolated_tensor = torch.complex(interpolated_real, interpolated_imaginary)

        return interpolated_tensor


class DownSampleBatch(torch.nn.Module):
    def __init__(self, dx: int, dy: int, original_dims: bool = False):

        super(DownSampleBatch, self).__init__()
        self.dx = dx
        self.dy = dy
        self.original_dims = original_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5  # (N,5,2,32,492)
        if self.dx == 1 and self.dy == 1:
            return x
        ds_data = x[:, :, :, :: self.dy, :: self.dx]
        if self.original_dims:
            return self._interpolate(ds_data)
        return ds_data

    def _interpolate(self, data):
        # Separate real and imaginary parts
        real_part = data.real
        imaginary_part = data.imag

        # Calculate the new size based on scale factors
        new_height = int(real_part.shape[-2] * self.dy)
        new_width = int(real_part.shape[-1] * self.dx)

        # Perform interpolation on both real and imaginary parts
        interpolated_real = F.interpolate(
            real_part,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        interpolated_imaginary = F.interpolate(
            imaginary_part,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

        # Combine interpolated real and imaginary parts back into a complex tensor
        interpolated_tensor = torch.complex(interpolated_real, interpolated_imaginary)

        return interpolated_tensor
