# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from function import calc_mean_std, mean_variance_norm
from models import MSP
from torch.nn import init
from models import BaseModel
import kornia.augmentation as K

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

# Aesthetic discriminator
class AesDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(AesDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        # Construct three discriminator models
        self.models = nn.ModuleList()
        self.score_models = nn.ModuleList()
        for i in range(3):
            self.models.append(
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512)
                )
            )
            self.score_models.append(
                nn.Sequential(
                    nn.Conv2d(512, 1, 3, padding=1)
                )
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    # Compute the MSE between model output and scalar gt
    def compute_loss(self, x, gt):
        _, outputs = self.forward(x)
        
        loss = sum([torch.mean((out - gt) ** 2) for out in outputs])
        return loss

    def forward(self, x):
        outputs = []
        feats = []
        for i in range(len(self.models)):
            feats.append(self.models[i](x))
            outputs.append(self.score_models[i](self.models[i](x)))
            x = self.downsample(x)
            
        self.upsample = nn.Upsample(size=(feats[0].size()[2],feats[0].size()[3]), mode='nearest')
        feat = feats[0]
        for i in range(1,len(feats)):
            feat += self.upsample(feats[i])
        
        return feat, outputs


# Aesthetic-aware style-attention (AesSA) module
class AesSA(nn.Module):
    def __init__(self, in_planes):
        super(AesSA, self).__init__()
        self.a = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.b = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.c = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.d = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.e = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.o1 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv1 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.out_conv2 = nn.Conv2d(in_planes, in_planes, (1, 1))
    def forward(self, content, style, aesthetic_feats):  
        if aesthetic_feats != None:
            A = self.a(aesthetic_feats)
        else:
            A = self.a(style)
            
        B = self.b(style) 
        b, c, h, w = A.size()
        A = A.view(b, -1, w * h)   # C x HsWs
        b, c, h, w = B.size()
        B = B.view(b, -1, w * h).permute(0, 2, 1)    # HsWs x C
        AS = torch.bmm(A, B)    # C x C
        AS = self.sm(AS)        # aesthetic attention map

        C = self.c(style)
        b, c, h, w = C.size()
        C = C.view(b, -1, w * h)   # C x HsWs
        astyle = torch.bmm(AS, C)     # C x HsWs

        astyle = astyle.view(b, c, h, w)
        astyle = self.out_conv1(astyle)
        astyle += style

        O1 = self.o1(mean_variance_norm(astyle))
        O1 = O1.view(b, -1, w * h)   # C x HsWs
        
        
        D = self.d(mean_variance_norm(content))
        b, c, h, w = D.size()
        D = D.view(b, -1, w * h).permute(0, 2, 1)   # HcWc x C
        
        S = torch.bmm(D, O1)    # HcWc x HsWs
        S = self.sm(S)          # style attention map
        
        
        E = self.e(astyle)
        b, c, h, w = E.size()
        E = E.view(b, -1, w * h)    # C x HsWs
        O = torch.bmm(E, S.permute(0, 2, 1))   # C x HcWc
        
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv2(O)
        O += content
        return O
    
    

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.AesSA_4_1 = AesSA(in_planes = in_planes)
        self.AesSA_5_1 = AesSA(in_planes = in_planes)
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1, content5_1, style5_1, aesthetic_feats=None):
        self.upsample_content4_1 = nn.Upsample(size=(content4_1.size()[2],content4_1.size()[3]), mode='nearest')
        self.upsample_style4_1 = nn.Upsample(size=(style4_1.size()[2],style4_1.size()[3]), mode='nearest')
        self.upsample_style5_1 = nn.Upsample(size=(style5_1.size()[2],style5_1.size()[3]), mode='nearest')
        if aesthetic_feats != None:
            return self.merge_conv(self.merge_conv_pad(self.AesSA_4_1(content4_1, style4_1, self.upsample_style4_1(aesthetic_feats)) + self.upsample_content4_1(self.AesSA_5_1(content5_1, style5_1, self.upsample_style5_1(aesthetic_feats)))))
        else:
            return self.merge_conv(self.merge_conv_pad(self.AesSA_4_1(content4_1, style4_1, aesthetic_feats) + self.upsample_content4_1(self.AesSA_5_1(content5_1, style5_1, aesthetic_feats))))
            

class Net(nn.Module):
    def __init__(self, encoder, decoder, discriminator, args):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        self.transform = Transform(in_planes = 512)
        self.decoder = decoder
        self.discriminator = discriminator
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.mse_loss = nn.MSELoss()
        self.args=args
        #**************************************************
        self.gpu_ids = [0]
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.nce_layers=[0,1,2,3]
        self.patch_sampler = K.RandomResizedCrop((256, 256), scale=(0.8, 1.0), ratio=(0.75, 1.33)).to(self.device)
        self.nce_loss = MSP.InfoNCELoss(self.args.temperature, self.args.hypersphere_dim,
                                       self.args.queue_size).to(self.device)
        self.style_vgg = MSP.vgg
        self.style_vgg.load_state_dict(torch.load('models/style_vgg.pth'))
        self.style_vgg = nn.Sequential(*list(self.style_vgg.children()))
        self.netD = MSP.StyleExtractor(self.style_vgg, self.gpu_ids)
        self.netP_style = MSP.Projector(self.gpu_ids)
        init_net(self.netD, 'normal', 0.02, self.gpu_ids)
        init_net(self.netP_style, 'normal', 0.02, self.gpu_ids)
        #**************************************************
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        #**********************************************************8
    # def optimize_parameters(self):
    #     # forward
    #     self.forward()
    #
    #     # update MSP
    #     if self.opt.lambda_NCE_D > 0.0:
    #         self.set_requires_grad([self.netD, self.netP_style], True)
    #         self.set_requires_grad([self.decoder, self.discriminator], False)
    #         self.optimizer_D_NCE.zero_grad()
    #         self.loss_NCE_D = self.backward_D_NCEloss()
    #         self.loss_NCE_D.backward(retain_graph=True)
    #         self.optimizer_D_NCE.step()
    #
    # #*********************************************************88


    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 features from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # content loss
    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))
#********************************************************************************************
    def backward_D_NCEloss(self, real_style):
        """
        Calculate NCE loss for the discriminator
        """
        # query_A = query_B =0.0
        #real_A = self.netD(self.patch_sampler(self.real_A), self.nce_layers)
        real_B = self.netD(self.patch_sampler(real_style), self.nce_layers)
        #real_Ax = self.netD(self.patch_sampler(self.real_A), self.nce_layers)
        real_Bx = self.netD(self.patch_sampler(real_style), self.nce_layers)

        #query_A = self.netP_style(real_A, self.nce_layers)
        query_B = self.netP_style(real_B, self.nce_layers)
        #query_Ax = self.netP_style(real_Ax, self.nce_layers)
        query_Bx = self.netP_style(real_Bx, self.nce_layers)

        num = 0
        loss_D_cont_A = 0
        loss_D_cont_B = 0
        for x in self.nce_layers:
            # self.nce_loss.dequeue_and_enqueue(query_A[num], 'real_A{:d}'.format(x))
            self.nce_loss.dequeue_and_enqueue(query_B[num], 'real_B{:d}'.format(x))
            # loss_D_cont_A += self.nce_loss(query_A[num], query_Ax[num], 'real_B{:d}'.format(x))
            loss_D_cont_B += self.nce_loss(query_B[num], query_Bx[num], 'real_B{:d}'.format(x))
            num += 1

        loss_NCE_D = (loss_D_cont_A + loss_D_cont_B) * 0.5
        return loss_NCE_D

#*****************************************************************************************8
    # style loss
    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    # style->Is   content->Ic
    def forward(self, content, style, aesthetic=False):
        #Fs
        style_feats = self.encode_with_intermediate(style)
        #Fc
        content_feats = self.encode_with_intermediate(content)
        if aesthetic:
            aesthetic_s_feats, _ = self.discriminator(style)
            stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4], aesthetic_s_feats)
        else:
            # aesSA MODULE
            stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
            # Ics
        g_t = self.decoder(stylized)
        # 第二个vgg encoder out
        g_t_feats = self.encode_with_intermediate(g_t)
        #print("g_t_feats[4]",g_t_feats[4])
        # content loss
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm = True) + self.calc_content_loss(g_t_feats[4], content_feats[4], norm = True)
        # style loss
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        # adversarial loss
        loss_gan_d = self.discriminator.compute_loss(style, 1) + self.discriminator.compute_loss(g_t.detach(), 0)
        loss_gan_g = self.discriminator.compute_loss(g_t, 1)
                
        #else:    # other losses in stage I
        # identity loss
        Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))

        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i], style_feats[i])
        loss_aesthetic = 0

        l_identity = 50 * l_identity1 + l_identity2

        # update MSP
        # if self.opt.lambda_NCE_D > 0.0:
        # SHeep=BaseModel()
        # SHeep.set_requires_grad([self.netD, self.netP_style], True)
        # SHeep.set_requires_grad([self.decoder, self.discriminator], False)
        # self.optimizer_D_NCE.zero_grad()
        # Comparative learning loss
        self.loss_NCE_D = self.backward_D_NCEloss(g_t.detach())
        # self.loss_NCE_D.backward(retain_graph=True)
        # self.optimizer_D_NCE.step()

    #*********************************************************88

        return self.loss_NCE_D, g_t, loss_c, loss_s, loss_gan_d, loss_gan_g, l_identity, loss_aesthetic

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, init_type, init_gain=init_gain)
    return net
