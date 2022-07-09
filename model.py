import torch.nn as nn
import math

class Encoder_cycle_noRes(nn.Module):
    def __init__(self, im_size, input_nc=3):
        super(Encoder_cycle_noRes, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 16, 7),
                    nn.InstanceNorm2d(16),
                    nn.ReLU(inplace=True) ]

        iterations = int(math.log2(im_size) - 4)
        # Downsampling
        in_features = 16
        out_features = in_features*2
        for _ in range(iterations):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Decoder_cycle(nn.Module):
    def __init__(self, input_nc=64, output_nc=3):
        super(Decoder_cycle, self).__init__()

        in_features = input_nc
        model = []

        iterations = int(math.log2(input_nc) - 4)
        out_features = in_features // 2
        for _ in range(iterations):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(16, output_nc, 7)]  # removed tanh activation

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(self, im_size=128, input_nc=3):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder_cycle_noRes(im_size, input_nc)
        self.decoder = Decoder_cycle(input_nc=128, output_nc=3)
    def forward(self, x):
        return self.decoder(self.encoder(x))
