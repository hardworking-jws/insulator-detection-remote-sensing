import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.01)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.01)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class SingleBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.block1 = BasicBlock(channels)
        self.block2 = BasicBlock(channels)
        self.block3 = BasicBlock(channels)
        self.block4 = BasicBlock(channels)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return out


class Transition(nn.Module):

    def __init__(self, in_channels, out_channels, is_down_sample):
        super(Transition, self).__init__()
        if is_down_sample:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                   stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.in_channel = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.other_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                        stride=1, bias=False)
            self.other_bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               stride=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.in_channel != self.out_channels:
            residual = self.other_conv(residual)
            residual = self.other_bn(residual)

        out = out + residual
        out = self.relu(out)

        return out


class HR_Net(nn.Module):
    def __init__(self, channel_num, class_num):
        super(HR_Net, self).__init__()
        self.n_classes = class_num
        self.n_channels = 3
        self.channel_num = channel_num
        self.relu = nn.ReLU(inplace=False)
        # TODO 修改进入通道数
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.01)
        self.bottleneck1 = Bottleneck(64, 256)
        self.bottleneck2 = Bottleneck(256, 256)
        self.bottleneck3 = Bottleneck(256, 256)
        self.bottleneck4 = Bottleneck(256, 256)

        self.transition1_A_to_B = Transition(256, 2 * channel_num, True)
        self.transition1_A_to_A = Transition(256, channel_num, False)
        self.stage2_blockA = SingleBlock(channel_num)
        self.stage2_blockB = SingleBlock(2 * channel_num)

        self.s12_A_to_B_conv = nn.Conv2d(in_channels=channel_num, out_channels=2 * channel_num, kernel_size=3, stride=2,
                                         padding=1, bias=False)
        self.s12_A_to_B_bn = nn.BatchNorm2d(2 * channel_num, momentum=0.01)
        self.s12_B_to_A_conv = nn.Conv2d(in_channels=2 * channel_num, out_channels=channel_num, kernel_size=1, stride=1,
                                         padding=0, bias=False)
        self.s12_B_to_A_bn = nn.BatchNorm2d(channel_num, momentum=0.01)
        self.s12_B_to_A_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.transition2_B_to_C = Transition(2 * channel_num, 4 * channel_num, True)
        self.stage3_blockA = SingleBlock(channel_num)
        self.stage3_blockB = SingleBlock(2 * channel_num)
        self.stage3_blockC = SingleBlock(4 * channel_num)

        self.s23_A_to_B_conv = nn.Conv2d(in_channels=channel_num, out_channels=2 * channel_num, kernel_size=3, stride=2,
                                         padding=1, bias=False)
        self.s23_A_to_B_bn = nn.BatchNorm2d(2 * channel_num, momentum=0.01)
        self.s23_A_to_C_conv1 = nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=3, stride=2,
                                          padding=1, bias=False)
        self.s23_A_to_C_bn1 = nn.BatchNorm2d(channel_num, momentum=0.01)
        self.s23_A_to_C_conv2 = nn.Conv2d(in_channels=channel_num, out_channels=4 * channel_num, kernel_size=3,
                                          stride=2, padding=1, bias=False)
        self.s23_A_to_C_bn2 = nn.BatchNorm2d(4 * channel_num, momentum=0.01)
        self.s23_B_to_A_conv = nn.Conv2d(in_channels=2 * channel_num, out_channels=channel_num, kernel_size=1, stride=1,
                                         padding=0, bias=False)
        self.s23_B_to_A_bn = nn.BatchNorm2d(channel_num, momentum=0.01)
        self.s23_B_to_A_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.s23_B_to_C_conv = nn.Conv2d(in_channels=2 * channel_num, out_channels=4 * channel_num, kernel_size=3,
                                         stride=2, padding=1, bias=False)
        self.s23_B_to_C_bn = nn.BatchNorm2d(4 * channel_num, momentum=0.01)
        self.s23_C_to_B_conv = nn.Conv2d(in_channels=4 * channel_num, out_channels=2 * channel_num, kernel_size=1,
                                         stride=1, padding=0, bias=False)
        self.s23_C_to_B_bn = nn.BatchNorm2d(2 * channel_num, momentum=0.01)
        self.s23_C_to_B_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.s23_C_to_A_conv = nn.Conv2d(in_channels=4 * channel_num, out_channels=channel_num, kernel_size=1, stride=1,
                                         padding=0, bias=False)
        self.s23_C_to_A_bn = nn.BatchNorm2d(channel_num, momentum=0.01)
        self.s23_C_to_A_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.transition3_C_to_D = Transition(4 * channel_num, 8 * channel_num, True)
        self.stage4_blockA = SingleBlock(channel_num)
        self.stage4_blockB = SingleBlock(2 * channel_num)
        self.stage4_blockC = SingleBlock(4 * channel_num)
        self.stage4_blockD = SingleBlock(8 * channel_num)

        self.s34_B_to_A_conv = nn.Conv2d(in_channels=2 * channel_num, out_channels=channel_num, kernel_size=1, stride=1,
                                         padding=0, bias=False)
        self.s34_B_to_A_bn = nn.BatchNorm2d(channel_num, momentum=0.01)
        self.s34_B_to_A_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.s34_C_to_A_conv = nn.Conv2d(in_channels=4 * channel_num, out_channels=channel_num, kernel_size=1, stride=1,
                                         padding=0, bias=False)
        self.s34_C_to_A_bn = nn.BatchNorm2d(channel_num, momentum=0.01)
        self.s34_C_to_A_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.s34_D_to_A_conv = nn.Conv2d(in_channels=8 * channel_num, out_channels=channel_num, kernel_size=1, stride=1,
                                         padding=0, bias=False)
        self.s34_D_to_A_bn = nn.BatchNorm2d(channel_num, momentum=0.01)
        self.s34_D_to_A_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # self.final_Layer = nn.Conv2d(in_channels=channel_num, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.final_Layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(
                in_channels=4*channel_num,
                out_channels=4*channel_num,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(4*channel_num, momentum=0.01),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=4*channel_num,
                out_channels=self.n_classes,
                kernel_size=1,
                stride=1,
                padding=0)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)

        stage2_A_in = self.transition1_A_to_A(x)
        stage2_B_in = self.transition1_A_to_B(x)
        stage2_A_middle = self.stage2_blockA(stage2_A_in)
        stage2_B_middle = self.stage2_blockB(stage2_B_in)
        stage2_A_out = self.relu(stage2_A_middle + self.s12_B_to_A_upsample(
            self.s12_B_to_A_bn(self.s12_B_to_A_conv(stage2_B_middle))))
        stage2_B_out = self.relu(stage2_B_middle + self.s12_A_to_B_bn(
            self.s12_A_to_B_conv(stage2_A_middle)))

        stage3_A_in = stage2_A_out
        stage3_B_in = stage2_B_out
        stage3_C_in = self.transition2_B_to_C(stage3_B_in)
        stage3_A_middle = self.stage3_blockA(stage3_A_in)
        stage3_B_middle = self.stage3_blockB(stage3_B_in)
        stage3_C_middle = self.stage3_blockC(stage3_C_in)
        stage3_A_out = self.relu(
            stage3_A_middle + self.s23_B_to_A_upsample(self.s23_B_to_A_bn(self.s23_B_to_A_conv(stage3_B_middle)))
            + self.s23_C_to_A_upsample(self.s23_C_to_A_bn(self.s23_C_to_A_conv(stage3_C_middle))))
        stage3_B_out = self.relu(stage3_B_middle + self.s23_A_to_B_bn(self.s23_A_to_B_conv(stage3_A_middle)) +
                                 self.s23_C_to_B_upsample(self.s23_C_to_B_bn(self.s23_C_to_B_conv(stage3_C_middle))))
        stage3_C_out = self.relu(stage3_C_middle + self.s23_A_to_C_bn2(
            self.s23_A_to_C_conv2(self.relu(self.s23_A_to_C_bn1(self.s23_A_to_C_conv1(stage3_A_middle))))) +
                                 self.s23_B_to_C_bn(self.s23_B_to_C_conv(stage3_B_middle)))

        stage4_A_in = stage3_A_out
        stage4_B_in = stage3_B_out
        stage4_C_in = stage3_C_out
        stage4_D_in = self.transition3_C_to_D(stage4_C_in)
        stage4_A_middle = self.stage4_blockA(stage4_A_in)
        stage4_B_middle = self.stage4_blockB(stage4_B_in)
        stage4_C_middle = self.stage4_blockC(stage4_C_in)
        stage4_D_middle = self.stage4_blockD(stage4_D_in)
        # stage4_out = self.relu(
        #     stage4_A_middle + self.s34_B_to_A_upsample(self.s34_B_to_A_bn(self.s34_B_to_A_conv(stage4_B_middle)))
        #     + self.s34_C_to_A_upsample(self.s34_C_to_A_bn(self.s34_C_to_A_conv(stage4_C_middle))) +
        #     self.s34_D_to_A_upsample(self.s34_D_to_A_bn(self.s34_D_to_A_conv(stage4_D_middle))))
        final_in = torch.cat([stage4_A_middle, self.s34_B_to_A_upsample(self.s34_B_to_A_bn(self.s34_B_to_A_conv(stage4_B_middle))),
                              self.s34_C_to_A_upsample(self.s34_C_to_A_bn(self.s34_C_to_A_conv(stage4_C_middle))),
                              self.s34_D_to_A_upsample(self.s34_D_to_A_bn(self.s34_D_to_A_conv(stage4_D_middle)))], 1)
        final_out = self.final_Layer(final_in)
        return final_out

    def load_pretrained_model(self, pretrained_model_path):
        pretrained_dict = torch.load(pretrained_model_path)
        print("从" + pretrained_model_path + "中加载预训练模型...，hr_seg_head")
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


if __name__ == '__main__':
    x = torch.rand(8, 1, 128, 128).cuda()
    net = HR_Net(40, 8)
    print(net)
