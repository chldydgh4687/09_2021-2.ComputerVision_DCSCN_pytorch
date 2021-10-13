from torch import nn
import torchvision
import torch
import numpy as np

# dcscn_L12_F196to48_NIN_A64_PS_R1F32
# B 32
class DCSCN(nn.Module):

    def __init__ (self, kernel_size=3, n_channels=64, scaling_factor=2):
        super(DCSCN, self).__init__()

        # filter calculation
        # layer = 12
        # max_filter = 196
        # min_filter = 48
        # for i in range(layer):
        #     x1 = i / float(layer - 1)
        #     y1 = pow(x1, 1.0 / 1.5)
        #     output_feature_num = int((max_filter - min_filter) * (1 - y1) + min_filter)
        #     print(output_feature_num)

        # ACTIVE FUNC & DROPOUT

        self.drop = torch.nn.Dropout(p=0.8)
        self.prelu = torch.nn.PReLU()

        # FEATURE EXTRACTION LEVEL

        self.conv1 = torch.nn.Conv2d(1,196,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv2 = torch.nn.Conv2d(196,166,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv3 = torch.nn.Conv2d(166,148,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv4 = torch.nn.Conv2d(148,133,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv5 = torch.nn.Conv2d(133,120,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv6 = torch.nn.Conv2d(120,108,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv7 = torch.nn.Conv2d(108,97,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv8 = torch.nn.Conv2d(97,86,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv9 = torch.nn.Conv2d(86,76,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv10 = torch.nn.Conv2d(76,66,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv11 = torch.nn.Conv2d(66,57,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.conv12 = torch.nn.Conv2d(57,48,3,stride=1,padding=1,padding_mode="replicate",bias=True)

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        torch.nn.init.kaiming_normal_(self.conv6.weight)
        torch.nn.init.kaiming_normal_(self.conv7.weight)
        torch.nn.init.kaiming_normal_(self.conv8.weight)
        torch.nn.init.kaiming_normal_(self.conv9.weight)
        torch.nn.init.kaiming_normal_(self.conv10.weight)
        torch.nn.init.kaiming_normal_(self.conv11.weight)
        torch.nn.init.kaiming_normal_(self.conv12.weight)

        # RECONSTRUCTION NETWORK LEVEL

        self.A1 = torch.nn.Conv2d(1301,64,1,stride=1,bias=True)
        self.B1 = torch.nn.Conv2d(1301,32,1,stride=1,bias=True)
        self.B2 = torch.nn.Conv2d(32,32,3,stride=1,padding=1,padding_mode="replicate",bias=True)

        torch.nn.init.kaiming_normal_(self.A1.weight)
        torch.nn.init.kaiming_normal_(self.B1.weight)
        torch.nn.init.kaiming_normal_(self.B2.weight)

        # Upsampled layer
        self.upconv = torch.nn.Conv2d(96,2*2*96,3,stride=1,padding=1,padding_mode="replicate",bias=True)
        self.pixelshufflerlayer = torch.nn.PixelShuffle(2)

        torch.nn.init.kaiming_normal_(self.upconv.weight)

        self.reconv = torch.nn.Conv2d(96,1,3,stride=1,padding=1,padding_mode="replicate",bias=False)

        torch.nn.init.kaiming_normal_(self.reconv.weight)

    def forward(self, lr):

        #Feature Update
        output = self.drop(self.prelu(self.conv1(lr)))
        s1 = output
        output = self.drop(self.prelu(self.conv2(output)))
        s2 = output
        output = self.drop(self.prelu(self.conv3(output)))
        s3 = output
        output = self.drop(self.prelu(self.conv4(output)))
        s4 = output
        output = self.drop(self.prelu(self.conv5(output)))
        s5 = output
        output = self.drop(self.prelu(self.conv6(output)))
        s6 = output
        output = self.drop(self.prelu(self.conv7(output)))
        s7 = output
        output = self.drop(self.prelu(self.conv8(output)))
        s8 = output
        output = self.drop(self.prelu(self.conv9(output)))
        s9 = output
        output = self.drop(self.prelu(self.conv10(output)))
        s10 = output
        output = self.drop(self.prelu(self.conv11(output)))
        s11 = output
        output = self.drop(self.prelu(self.conv12(output)))
        s12 = output
        output = torch.cat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12], dim = 1)

        # Reconstruction Update

        a1_out = self.drop(self.prelu(self.A1(output)))
        b1_out = self.drop(self.prelu(self.B1(output)))
        b2_out = self.drop(self.prelu(self.B2(b1_out)))
        output = torch.cat([a1_out, b2_out], dim = 1)
        # transposed

        up_out = self.pixelshufflerlayer(self.upconv(output))
        re_out = self.reconv(up_out)
        return re_out


# if __name__ == "__main__" :
#     model = DCSCN()
#     a = torch.randn([20,1,32,32])
#     print(a.shape)
#     print(model(a).shape)
