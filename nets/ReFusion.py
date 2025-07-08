import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable
import numbers

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.groups = self.conv.groups
        self.kernel_size = self.conv.kernel_size
        self.weight=self.conv.weight
        self.bias=self.conv.bias

        self.weight_meta=to_var(self.conv.weight.data, requires_grad=True)
        if self.conv.bias is not None:
            self.bias_meta=to_var(self.conv.bias.data, requires_grad=True)
        else:
            self.bias_meta=None

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
    def device_check(self,x):
        for v in [self.weight_meta,self.bias_meta]:
            if v is not None:
                if x.is_cuda:
                    v.cuda()

    def forward(self, x,meta=False):
        if meta: 
            self.device_check(x)
            return F.conv2d(x, self.weight_meta, self.bias_meta, self.stride, self.padding, self.dilation, self.groups)
        else:
            return self.conv(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

        self.weight_meta=to_var(self.weight.data, requires_grad=True)
        self.bias_meta=to_var(self.bias.data, requires_grad=True)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
    def device_check(self,x):
        for v in [self.weight_meta,self.bias_meta]:
            if x.is_cuda:
                v.cuda()

    def forward(self, x,meta=False):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if meta:
            self.device_check(x)
            return (x - mu) / torch.sqrt(sigma+1e-5)* self.weight_meta + self.bias_meta
        else:    
            return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)
    def forward(self, x,meta=False):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x),meta=meta), h, w)

## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module): 
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in =MetaConv2d(in_channels=dim,out_channels=hidden_features*2,kernel_size=1,bias=bias)

        self.dwconv = MetaConv2d(in_channels=hidden_features*2,out_channels=hidden_features*2,stride=1,padding=1,kernel_size=3,groups=hidden_features*2,bias=bias)


        self.project_out = MetaConv2d(in_channels=hidden_features,out_channels=dim,kernel_size=1,bias=bias)

    def forward(self, x,meta=False):
        x = self.project_in(x,meta=meta)
        x1, x2 = self.dwconv(x,meta=meta).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x,meta=meta)
        return x

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_meta=to_var(self.temperature.data)
        self.qkv =MetaConv2d(in_channels=dim,out_channels=dim*3,kernel_size=1,bias=bias)

        self.qkv_dwconv = MetaConv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = MetaConv2d(dim, dim, kernel_size=1, bias=bias)

    def named_leaves(self):
        return [('temperature', self.temperature)]
    
    def device_check(self,x):
        for v in [self.temperature_meta]:
            if v is not None:
                if x.is_cuda:
                    v.cuda()

    def forward(self, x,meta=False):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x,meta=meta),meta=meta)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        if meta:
            self.device_check(x)
            attn = (q @ k.transpose(-2, -1)) * self.temperature_meta
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out,meta=meta)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, meta=False):
        x = x + self.attn(self.norm1(x,meta=meta),meta=meta)
        x = x + self.ffn(self.norm2(x,meta=meta),meta=meta)

        return x

class MetaConvUnit(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1):
        super(MetaConvUnit, self).__init__()
        p = kernel_size//2
        self.conv=MetaConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p,bias=True)
        self.act=MetaPReLU()
    def forward(self, input,meta=False):
        x=self.conv(input,meta=meta)
        x=self.act(x,meta=meta)
        return x    

class MetaPReLU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        super(MetaPReLU, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))
        self.weight_meta=to_var(self.weight.data, requires_grad=True)
    
    def named_leaves(self):
        return [('weight', self.weight)]
    
    def device_check(self,x):
        for v in [self.weight_meta]:
            if v is not None:
                if x.is_cuda:
                    v.cuda()

    def forward(self, x,meta=False):
        
        if meta:
            self.device_check(x)
            return F.prelu(x, self.weight_meta)
        else:
            return F.prelu(x, self.weight)
          

class AFM(nn.Module):
    def __init__(self,dim=16, num_heads=8, bias=False):
        super(AFM, self).__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature1_meta=to_var(self.temperature1.data)

        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2_meta=to_var(self.temperature2.data)

        self.qkv1 =MetaConv2d(in_channels=dim,out_channels=dim*3,kernel_size=1,bias=bias)
        self.qkv1_dwconv = MetaConv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        
        self.qkv2 =MetaConv2d(in_channels=dim,out_channels=dim*3,kernel_size=1,bias=bias)
        self.qkv2_dwconv = MetaConv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)

        self.project_mid1 = MetaConv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_mid2 = MetaConv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_E = MetaConv2d(in_channels=dim*2,out_channels=dim*8,kernel_size=1,bias=bias)
        self.project_E_d = MetaConv2d(in_channels=dim*8,out_channels=dim*8,stride=1,padding=1,kernel_size=3,groups=dim*8,bias=bias)

        self.project_E1 = MetaConv2d(in_channels=dim,out_channels=dim*4,kernel_size=1,bias=bias)
        self.project_E1_d = MetaConv2d(in_channels=dim*4,out_channels=dim*4,stride=1,padding=1,kernel_size=3,groups=dim*4,bias=bias)

        self.project_E2 = MetaConv2d(in_channels=dim,out_channels=dim*4,kernel_size=1,bias=bias)
        self.project_E2_d = MetaConv2d(in_channels=dim*4,out_channels=dim*4,stride=1,padding=1,kernel_size=3,groups=dim*4,bias=bias)
        
        self.project_out1 = MetaConv2d(dim*4, dim*2, kernel_size=1, bias=bias)
        self.project_out2 = MetaConv2d(dim*4, dim*2, kernel_size=1, bias=bias)
    def named_leaves(self):
        return [('temperature1', self.temperature1),('temperature2', self.temperature2)]
    
    def device_check(self,x):
        for v in [self.temperature1_meta,self.temperature2_meta]:
            if v is not None:
                if x.is_cuda:
                    v.cuda()
    
    def forward(self, x1, x2, meta=False):
        b, c, h, w = x1.shape

        qkv1 = self.qkv1_dwconv(self.qkv1(x1,meta=meta),meta=meta)
        q1, k1, v1 = qkv1.chunk(3, dim=1)

        qkv2 = self.qkv2_dwconv(self.qkv2(x2,meta=meta),meta=meta)
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)',head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)',head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)',head=self.num_heads)
        
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)',head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)',head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)',head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        if meta:
            self.device_check(x1)
            attn1 = (q2 @ k1.transpose(-2, -1)) * self.temperature1_meta
            attn2 = (q1 @ k2.transpose(-2, -1)) * self.temperature2_meta
        else:
            attn1 = (q2 @ k1.transpose(-2, -1)) * self.temperature1
            attn2 = (q1 @ k2.transpose(-2, -1)) * self.temperature2

        out1= (attn1.softmax(dim=-1) @ v1)
        out2= (attn2.softmax(dim=-1) @ v2)

        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        
        x1=x1+self.project_mid1(out1,meta=meta)
        x2=x2+self.project_mid2(out2,meta=meta)
        out=torch.cat([x1,x2],1)

        g1, g2 = self.project_E_d(self.project_E(out,meta=meta),meta=meta).chunk(2, dim=1)
        x1=F.gelu(g1)*self.project_E1_d(self.project_E1(x1,meta=meta),meta=meta)
        x2=F.gelu(g2)*self.project_E2_d(self.project_E2(x2,meta=meta),meta=meta)
        return out+self.project_out1(x1,meta=meta)+self.project_out2(x2,meta=meta)

class ReFusion(nn.Module):
    def __init__(self,
                 dim=16,
                 num_blocks=2,
                 heads=8,
                 ffn_expansion_factor=2,
                 bias=False,):
        super().__init__()
        self.project_in_1=MetaConv2d(in_channels=1,out_channels=dim,stride=1,padding=1,kernel_size=3,bias=bias)
        self.project_in_2=MetaConv2d(in_channels=1,out_channels=dim,stride=1,padding=1,kernel_size=3,bias=bias)

        self.encoder1 = nn.ModuleList([TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks)])

        self.encoder2 = nn.ModuleList([TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks)])
        
        self.FM=AFM(dim=dim,num_heads=heads, bias=bias)
        
        self.decoder = nn.ModuleList([TransformerBlock(dim=dim*2, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks)])

        self.project_out_0=MetaConv2d(in_channels=dim*2,out_channels=dim,stride=1,padding=1,kernel_size=3,bias=bias)
        self.act=nn.LeakyReLU()
        self.project_out_1=MetaConv2d(in_channels=dim,out_channels=1,stride=1,padding=1,kernel_size=3,bias=bias)

    def forward(self, x,meta=False):
        x1=x[:,:1,:,:]
        x2=x[:,1:,:,:]
        x1 = self.project_in_1(x1,meta=meta)
        x2 = self.project_in_2(x2,meta=meta)

        for layer in self.encoder1:
            x1 = layer(x1,meta=meta)
        for layer in self.encoder2:
            x2 = layer(x2,meta=meta)
        out=self.FM(x1,x2,meta=meta)
        for layer in self.decoder:
            out = layer(out,meta=meta)

        out=self.project_out_0(out,meta=meta)
        out=self.act(out)
        out=self.project_out_1(out,meta=meta)
        return nn.Sigmoid()(out)

class LPN(nn.Module):
    def __init__(self,
                 dim=16,
                 num_blocks=2,
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 ):

        super(LPN, self).__init__()

        self.patch_embed1=MetaConv2d(in_channels=1,out_channels=dim,stride=1,padding=1,kernel_size=3,bias=bias)
        self.patch_embed2=MetaConv2d(in_channels=1,out_channels=dim,stride=1,padding=1,kernel_size=3,bias=bias)

        self.encoder1 = nn.ModuleList([TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks)])
        self.encoder2 = nn.ModuleList([TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks)])        

        self.encoder3 = nn.ModuleList([TransformerBlock(dim=2*dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks)]) 
        
        self.out1=MetaConv2d(in_channels=int(2*dim),out_channels=int(dim),stride=1,padding=1,kernel_size=3,bias=bias)
        self.out2=MetaPReLU()
        self.out3=MetaConv2d(in_channels=int(dim),out_channels=2,stride=1,padding=1,kernel_size=3,bias=bias)

    def forward(self, x,meta=False):
        x1=x[:,:1,:,:]
        x2=x[:,1:,:,:]

        x1 = self.patch_embed1(x1,meta=meta)
        for subm in self.encoder1:
            x1 = subm(x1,meta=meta)
        x2 = self.patch_embed2(x2,meta=meta)
        for subm in self.encoder2:
            x2 = subm(x2,meta=meta)
        out=torch.cat((x1,x2),1)
        for subm in self.encoder3:
            out = subm(out,meta=meta)
        out=self.out1(out,meta=meta)
        out=self.out2(out)
        out=self.out3(out,meta=meta)

        return nn.Softmax(dim=1)(out)
    


def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(16,2,128,128).astype(np.float32)).cuda()
    model = ReFusion()
    model.cuda()
    y = model(x)
    print('output shape:', y.shape)

def count_parameter(net):
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.3fM" % (total/1e6))
    return total

if __name__ == '__main__':
    unit_test()
    count_parameter(ReFusion())

 

    
