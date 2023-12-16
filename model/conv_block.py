import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3,
                 stride = 1, padding = 1, bias = True, norm = "bnorm", relu = True) -> None:
        super().__init__()
        
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride,
                      padding = padding, bias = bias)
        )
        
        if(norm == "bnorm"):
            layers.append(nn.BatchNorm2d(out_channels))
        
        if(relu == True):
            layers.append(nn.ReLU(relu))
            
        self.conv_layer = nn.Sequential()
        for idx, layer in enumerate(layers):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), idx)
            self.conv_layer.add_module(layer_name, layer)
        self.init_param()
        
    def init_param(self):
        for module in self.modules():
            if(isinstance(module, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
                
            elif(isinstance(module, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        return self.conv_layer(x)