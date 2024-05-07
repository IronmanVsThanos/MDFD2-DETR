import torch
import torch.nn.functional as F
import torch.nn as nn



class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        # x → 1 16 512  这里按第二个维度也就是通道维度view进行分组，最后变成512 其实就是把每个组内的2个通道乘上宽高 2*W*H https://blog.csdn.net/weixin_44492824/article/details/124025689
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SFRM(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False,  # 是否使用PyTorch内置的GroupNorm，默认为False

                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        # scale_factor = abs(1-self.gn.gamma) + self.gn.beta
        normal_scale_factor = abs(1-self.gn.gamma) + self.gn.beta / sum(abs(1 - self.gn.gamma) + self.gn.beta)
        re_weights = gn_x * normal_scale_factor
        new_input = self.sigomid(re_weights + x)
        info_mask = new_input >= self.gate_treshold
        x_s = info_mask * x
        return x_s


class FDFM(nn.Module):
    def __init__(self, dim: int = 256, h: int = 64, w: int = 64):
        super().__init__()
        self.h = h
        self.w = w
        self.complex_weight = nn.Parameter(torch.randn(dim, self.h, self.w, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, C, H, W,  = x.shape  # 1 28 16 3
        # 第一个维度 (1): 表示批次大小，这里为1。
        # 第二个维度 (28): 表示输入张量的高度。
        # 第三个维度 (9): 表示变换后的宽度，对应输入宽度的一半加一（16//2 + 1）。
        # 第四个维度 (3): 表示通道数。
        # 这个输出表明在每个通道的每个位置上都有一个复数值，其中包含了傅里叶变换的实部和虚部。如果需要访问实部和虚部，可以使用索引 [..., :, :, 0] 和 [..., :, :, 1]。
        x_fft = torch.fft.rfft2(x, dim=(1), norm='ortho')
        # 在 PyTorch 中，torch.view_as_complex 是一个函数，
        # 用于将两个实部和虚部的实数张量视为一个复数张量。这个函数在处理复数张量时非常有用，特别是在涉及频域操作（如傅里叶变换）时。
        weight = torch.view_as_complex(self.complex_weight)
        # strat_idx_h =torch.randint(0, self.h - H + 1, (1,))
        # strat_idx_w =torch.randint(0, self.w - W + 1, (1,))
        # selected_weight = weight[:,strat_idx_h:strat_idx_h + H,strat_idx_w:strat_idx_w + W]
        _x = x_fft * weight
        # c = C.item()
        x_ifft = torch.fft.irfft2(_x, s=(C), dim=(1), norm='ortho')
        x = x_ifft + x
        return x


class CFM(nn.Module):
    def __init__(self, oup_channels: int, alpha: float = 1 / 2,  group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__()
        self.op_channel = oup_channels
        self.up_channel = up_channel = int(alpha * oup_channels)  # 计算上层通道数
        self.low_channel = low_channel = oup_channels - up_channel  # 计算下层通道数
        self.squeeze1 = nn.Conv2d(self.op_channel, up_channel, kernel_size=1, bias=False)  # 创建卷积层
        self.squeeze2 = nn.Conv2d(self.op_channel, low_channel, kernel_size=1, bias=False)  # 创建卷积层
        # 组卷积
        self.GWC1 = nn.Conv2d(up_channel, up_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        # 组卷积
        self.GWC2 = nn.Conv2d(low_channel, low_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        # 逐点卷积 其实就是kernel_size = 1的卷积
        self.PWC1 = nn.Conv2d(up_channel, up_channel, kernel_size=1, bias=False)  # 创建卷积层

        # 下层特征转换
        self.PWC2 = nn.Conv2d(low_channel, low_channel, kernel_size=1,bias=False)  # 创建卷积层
        self.GAP = nn.AdaptiveAvgPool2d(1)  # 创建自适应平均池化层

    def forward(self, x):
        x_sf_up = self.squeeze1(x)
        x_sf_down = self.squeeze2(x)

        y1 = self.PWC1(x_sf_up) + self.GWC2(x_sf_down)
        y2 = self.PWC2(x_sf_down) + self.PWC2(x_sf_down)
        x1 = self.GAP(y1)
        x2 = self.GAP(y2)
        x_new = F.softmax(torch.cat([x1, x2], dim=1), dim=1)
        out1, out2 = torch.split(x_new, x_new.size(1) // 2, dim=1)
        c1 = out1 * y1
        c2 = out2 * y2
        out = torch.cat([c1, c2], dim=1)
        return out


# 自定义 ScConv（Squeeze and Channel Reduction Convolution）模型
class SFCM(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 16, gate_treshold: float = 0.5, alpha: float = 1 / 2,
                 squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()  # 调用父类构造函数

        self.SFRM = SFRM(op_channel, group_num=group_num, gate_treshold=gate_treshold)  # 创建 SRU 层
        self.FDFM = FDFM(dim=int(op_channel/2+1), h=20, w=20)  # 创建 CRU 层
        self.CFM = CFM(op_channel, alpha=alpha, group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        xs = self.SFRM(x)
        xf = self.FDFM(xs)
        xc = self.CFM(xf)
        return xc


if __name__ == '__main__':
    x = torch.randn(1, 256, 64, 64)  # 创建随机输入张量
    model = SFCM(256)  # 创建 ScConv 模型
    print(model(x).shape)  # 打印模型输出的形状
