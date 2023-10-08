import torch
import torch.nn as nn
import torch.nn.functional as F



class NLBlockND_cross(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', bn_layer=True):
        super(NLBlockND_cross, self).__init__()

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2

        # Use 1D convolutions
        self.g = nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                nn.Conv1d(self.inter_channels, self.in_channels, kernel_size=1),
                nn.BatchNorm1d(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv1d(self.inter_channels, self.in_channels, kernel_size=1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(self.inter_channels * 2, 1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch, x_otherBranch):
        dtype = next(self.parameters()).dtype

        # Ensure that the inputs are of the same dtype as the module's weights
        x_thisBranch = x_thisBranch.to(dtype=dtype)
        x_otherBranch = x_otherBranch.to(dtype=dtype)
        # Reshape inputs
        x_thisBranch = x_thisBranch.permute(1, 2, 0)  # [32, 768, 197]
        x_otherBranch = x_otherBranch.permute(1, 2, 0)  # [32, 768, 197]

        batch_size = x_thisBranch.size(0)

        g_x = self.g(x_thisBranch).permute(0, 2, 1)
        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            # theta_x = theta_x.permute(0, 2, 1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)

            # elif self.mode == "concatenate":
        else:  # default as concatenate
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x_thisBranch

        return z

if __name__ == '__main__':
    import os

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda0 = torch.device('cuda:0')
    x = torch.rand(1,32,16, 8, 8).cuda(cuda0).float()
    y = torch.rand(1,32,16, 8, 8).cuda(cuda0).float()

    model = NLBlockND_cross(32)
    model = model.cuda()

    result = model(x,y)
    print(result.size())