import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class Double_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Double_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 
        for feature in features:
            self.downs.append(Double_Conv(in_channels, feature))
            in_channels = feature

        # Decoder 
        for feature in reversed(features): # go reverse
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(Double_Conv(feature*2, feature))
        
        # middle
        self.bottleneck = Double_Conv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse skip connections

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # if input is not divisible by 16 (ex: input 81 x 81 -> 40 x 40 -> 80 x 80 => cannot be concatenated)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # combine up with skip connection
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


# make sure it works 
def test_unet():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    #print(x.shape)
    #print(preds.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test_unet()