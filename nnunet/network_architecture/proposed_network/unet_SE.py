from typing import Tuple, Union
import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock, ResidualUnit, Convolution
from monai.networks.nets import ViT
from monai.networks.utils import normal_init

from nnunet.network_architecture.neural_network import SegmentationNetwork


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        
        self.conv = nn.Conv3d(c1, c2, kernel_size=1, stride=1, bias=False)
        self.In = nn.InstanceNorm3d(c2)
        self.leak = nn.LeakyReLU(negative_slope = 0.1, inplace = True)
 
    def forward(self, x):
        
        x = self.conv(x)
        x = self.In(x)
        x = self.leak(x)
        return x
    
        
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool3d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
        

class SE(nn.Module):
    def __init__(self, c1, c2, ratio=16):
        super(SE, self).__init__()
        #c*1*1
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class UNet_SE(SegmentationNetwork):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        do_ds:bool,
    ):
        

        super(UNet_SE, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.do_ds = do_ds
        
        self.down1 = ResidualUnit(
                                spatial_dims=3,
                                in_channels=self.in_channels,
                                out_channels=32,
                                adn_ordering="NA",
                                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True})
                            )
        
        self.down2 = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=32,
                out_channels=64,
                strides=[1,2,2],
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
            Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=64,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
            SE(c1=64, c2=64, ratio=4)
        )

        self.down3 = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=128,
                strides=[2,2,2],
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
            Convolution(
                spatial_dims=3,
                in_channels=128,
                out_channels=128,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ),
            SE(c1=128, c2=128, ratio=8)
        )

        self.down4 = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=128,
                out_channels=256,
                strides=[2,2,2],
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ),
            Convolution(
                spatial_dims=3,
                in_channels=256,
                out_channels=256,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ),
            SE(c1=256, c2=256, ratio=16)
        )

        self.down5 = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=256,
                out_channels=320,
                strides=[2,2,2],
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
            Convolution(
                spatial_dims=3,
                in_channels=320,
                out_channels=320,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ),
            SE(c1=320, c2=320, ratio=20)
        )

        self.bot = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=320,
                out_channels=320,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ),
            Convolution(
                spatial_dims=3,
                in_channels=320,
                out_channels=320,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            )

        )

        self.up1 = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=640,
                out_channels=320,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
            Convolution(
                spatial_dims=3,
                in_channels=320,
                out_channels=320,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ),
            SE(c1=320, c2=320, ratio=20)
        )

        self.up2 = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=512,
                out_channels=256,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
            Convolution(
                spatial_dims=3,
                in_channels=256,
                out_channels=256,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
            SE(c1=256, c2=256, ratio=16)
        )

        self.up3 = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=256,
                out_channels=128,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
            Convolution(
                spatial_dims=3,
                in_channels=128,
                out_channels=128,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
            SE(c1=128, c2=128, ratio=8)
        )

        self.up4 = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=128,
                out_channels=64,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
            Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=64,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
            SE(c1=64, c2=64, ratio=4)
        )

        self.up5 = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=32,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
            Convolution(
                spatial_dims=3,
                in_channels=32,
                out_channels=32,
                adn_ordering="NA",
                act=("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            ), 
        )


        self.t2 = Convolution(
                spatial_dims=3,
                in_channels=320,
                out_channels=256,
                kernel_size=[2,2,2],
                strides=[2,2,2],
                padding=0,
                output_padding=0,
                is_transposed=True,
                bias=False,
            )

        self.t3 = Convolution(
                spatial_dims=3,
                in_channels=256,
                out_channels=128,
                kernel_size=[2,2,2],
                strides=[2,2,2],
                padding=0,
                output_padding=0,
                is_transposed=True,
                bias=False,
            )

        self.t4 = Convolution(
                spatial_dims=3,
                in_channels=128,
                out_channels=64,
                kernel_size=[2,2,2],
                strides=[2,2,2],
                padding=0,
                output_padding=0,
                is_transposed=True,
                bias=False,
            )

        self.t5 = Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=32,
                kernel_size=[1,2,2],
                strides=[1,2,2],
                padding=0,
                output_padding=0,
                is_transposed=True,
                bias=False,
            )

        self.out1 = Convolution(
                spatial_dims=3,
                in_channels=320,
                out_channels=self.num_classes,
                kernel_size=[1,1,1],
                padding=[0,0,0],
                bias=False,
            )
        
        self.out2 = Convolution(
                spatial_dims=3,
                in_channels=256,
                out_channels=self.num_classes,
                kernel_size=[1,1,1],
                padding=[0,0,0],
                bias=False,
            )

        self.out3 = Convolution(
                spatial_dims=3,
                in_channels=128,
                out_channels=self.num_classes,
                kernel_size=[1,1,1],
                padding=[0,0,0],
                bias=False,
            )

        self.out4 = Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=self.num_classes,
                kernel_size=[1,1,1],
                padding=[0,0,0],
                bias=False,
            )

        self.out5 = Convolution(
                spatial_dims=3,
                in_channels=32,
                out_channels=self.num_classes,
                kernel_size=[1,1,1],
                padding=[0,0,0],
                bias=False,
            )

    def forward(self, x):
        x = self.down1(x)                                         ## 512*512*32
        s1 = self.down2(x)                                        ## 256*256*64
        s2 = self.down3(s1)                                       ## 128*128*128
        s3 = self.down4(s2)                                       ## 64*64*256
        s4 = self.down5(s3)                                       ## 32*32*320
        bottom = self.bot(s4)                                     ## 32*32*320

        out1 = self.up1(torch.cat([bottom, s4], 1))               ## 32*32*320
        out2 = self.up2(torch.cat([self.t2(out1), s3], 1))        ## 64*64*256
        out3 = self.up3(torch.cat([self.t3(out2), s2], 1))        ## 128*128*128
        out4 = self.up4(torch.cat([self.t4(out3), s1], 1))        ## 256*256*64
        out5 = self.up5(torch.cat([self.t5(out4), x], 1))         ## 512*512*32

        out1 = self.out1(out1)                                    ## 32*32*14
        out2 = self.out2(out2)                                    ## 64*64*14
        out3 = self.out3(out3)                                    ## 128*128*14
        out4 = self.out4(out4)                                    ## 256*256*14
        out5 = self.out5(out5)                                    ## 512*512*14
        if self.training or self.do_ds:
            return [out5, out4, out3, out2, out1]
        else:
            return out5


if __name__ == '__main__':
    input_t = torch.randn((2, 1, 40, 192, 224))
    model = UNet_SE(in_channels=1, num_classes=3, do_ds=True)
    output_t = model(input_t)
    a=0
