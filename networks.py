import jittor as jt
from jittor import Module
from jittor import nn

# Definition of normalization layer
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features

        self.norm = nn.InstanceNorm2d(num_features, affine=False)

        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
    
    def execute(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        out = self.norm(x)
        out = out * self.weight + self.bias
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class ConvBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(ConvBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def execute(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_type, padding_type, use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_type, padding_type, use_dropout)

    def build_conv_block(self, dim, norm_type, padding_type, use_dropout):
        conv_block = []
        conv_block += [ConvBlock(dim ,dim, 3, 1, 1, norm=norm_type, activation='relu', pad_type=padding_type)]
        conv_block += [ConvBlock(dim ,dim, 3, 1, 1, norm=norm_type, activation='none', pad_type=padding_type)]

        return nn.Sequential(*conv_block)

    def execute(self, x):
        out = x + self.conv_block(x)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=7, padding=3, stride=4):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)
        self.relu = nn.LeakyReLU(True)

    def execute(self, ten, out=False,t = False):
        # here we want to be able to take an intermediate output for reconstruction error
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = self.relu(ten)
        return ten

class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=4, padding=1, stride=2, output_padding=0, norelu=False):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double dimensions
        layers_list = []
        layers_list.append(nn.ConvTranspose(channel_in, channel_out, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding))
        layers_list.append(nn.BatchNorm2d(channel_out, momentum=0.9))
        if norelu == False:
            layers_list.append(nn.LeakyReLU(True))
        self.conv = nn.Sequential(*layers_list)

    def execute(self, ten):
        ten = self.conv(ten)
        return ten

class DrawingEncoder(nn.Module):
    def __init__(self, image_size, input_nc, norm_layer='bn', latent_dim=512):
        assert(latent_dim >= 0)
        super(DrawingEncoder, self).__init__()
        latent_size = int(image_size/32)
        longsize = 512*latent_size*latent_size

        layers_list = []
        layers_list.append(EncoderBlock(channel_in=input_nc, channel_out=32, kernel_size=4, padding=1, stride=2))
        
        dim_size = 32
        for i in range(4):
            layers_list.append(ResnetBlock(dim_size, padding_type='reflect', norm_type=norm_layer))  # 176 176  
            layers_list.append(EncoderBlock(channel_in=dim_size, channel_out=dim_size*2, kernel_size=4, padding=1, stride=2))  # 88 88
            dim_size *= 2

        layers_list.append(ResnetBlock(512, padding_type='reflect', norm_type=norm_layer))  # 176 176
        self.conv = nn.Sequential(*layers_list)

        self.fc_mu = nn.Sequential(nn.Linear(in_features=longsize, out_features=latent_dim))
        
    def execute(self, x):
        ten = self.conv(x)
        ten = ten.view(ten.size()[0],-1)
        mu = self.fc_mu(ten)
        return mu

class DrawingDecoder(nn.Module):
    def __init__(self, image_size, output_nc, latent_dim=512):
        super(DrawingDecoder, self).__init__()
        latent_size = int(image_size/32)
        self.latent_size = latent_size
        longsize = 512*latent_size*latent_size

        activation = nn.ReLU(True)
        padding_type='reflect'
        norm_layer='bn'
        #norm_layer=nn.BatchNorm2d

        self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=longsize))
        layers_list = []
        layers_list.append(ResnetBlock(512, padding_type=padding_type, norm_type=norm_layer))

        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(ResnetBlock(256, padding_type=padding_type, norm_type=norm_layer))

        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(ResnetBlock(256, padding_type=padding_type, norm_type=norm_layer))

        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(ResnetBlock(128, padding_type=padding_type, norm_type=norm_layer))

        layers_list.append(DecoderBlock(channel_in=128, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(ResnetBlock(64, padding_type=padding_type, norm_type=norm_layer))

        layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0))
        layers_list.append(ResnetBlock(64, padding_type=padding_type, norm_type=norm_layer))

        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(64,output_nc,kernel_size=5,padding=0))
        self.conv = nn.Sequential(*layers_list)

    def execute(self, ten):
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, self.latent_size, self.latent_size)
        ten = self.conv(ten)
        return ten

class GeometryEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=1, norm_layer=nn.InstanceNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GeometryEncoder, self).__init__()        
        activation = nn.ReLU()        

        model = [nn.ReflectionPad2d(3), nn.Conv(input_nc, ngf, 7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_type = 'in', padding_type=padding_type)]
        self.model = nn.Sequential(*model)
            
    def execute(self, input):
        return self.model(input)

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [ConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv(dim, style_dim, 1, 1, 0)]
        
        self.model = nn.Sequential(*self.model)
        
        self.output_dim = dim

    def execute(self, x):
        return self.model(x)

class Part_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer='adain', 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Part_Generator, self).__init__()        
        activation = nn.ReLU()

        model = []
        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_type='adain', padding_type=padding_type)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose(ngf * mult, int(ngf * mult / 2), 3, stride=2, padding=1, output_padding=1)]
            if norm_layer == 'adain':
                model += [AdaptiveInstanceNorm2d(int(ngf * mult / 2))]
            else:
                model += [nn.InstanceNorm2d(int(ngf * mult / 2))]
            model += [activation]
        model += [nn.ReflectionPad2d(3), nn.Conv(ngf, output_nc, 7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
  
        # style encoder
        self.enc_style = StyleEncoder(5, 3, 16, self.get_num_adain_params(self.model), norm='none', activ='relu', pad_type='reflect')

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean
                m.weight = std
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    def execute(self, image_content, image_style, adain_params = None):
        adain_params = self.enc_style(image_style)
        #print(adain_params.shape)
        self.assign_adain_params(adain_params, self.model)
        return self.model(image_content),adain_params

    def feature_execute(self, image_content, image_style):
        adain_params = self.enc_style(image_style)
        self.assign_adain_params(adain_params, self.model)
        for layer_id, layer in enumerate(self.model):
            image_content = layer(image_content)
            if layer_id == 15:
                return image_content

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm='in', 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU()        

        if norm == 'in':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv(input_nc, ngf, 7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_type = norm, padding_type=padding_type)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose(ngf * mult, int(ngf * mult / 2), 3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def execute(self, input):
        return self.model(input) 

