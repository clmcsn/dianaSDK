from torch._C import device
from torchvision.transforms import transforms
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)

from torchvision.datasets import CIFAR10, ImageNet

from brevitas.nn import QuantConv2d
from brevitas.nn.quant_activation import QuantReLU, QuantIdentity
from brevitas.nn.quant_bn import BatchNorm2dToQuantScaleBias
from brevitas.nn.quant_avg_pool import QuantAdaptiveAvgPool2d
from brevitas.aimc.nn.analog import AimcConv2d
from brevitas.nn.quant_linear import QuantLinear
from zoo.models.resnet.qresnet import ResidualLayer


class LayerParam():
    def __init__(self, weight, weight_scale, bias, bias_scale):
        self.weight = weight
        self.weight_scale = weight_scale
        self.bias = bias 
        self.bias_scale = bias_scale

class ModelInputClass():
    def __init__(self, size=None, ds=ImageNet, image=None, path=None):
        self.size = size
        self.ds = ds
        if image:
            self.image = image
        elif path:
            self.image = torch.load(path)
        else:
            self.image = self._gen_rand_input(self.size, self.ds)
        
    @staticmethod
    def _gen_rand_input(size=[1,3,224,224], ds=ImageNet):
        if ds==ImageNet:
            normalize = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        if ds==CIFAR10:
            normalize = transforms.Normalize(
                                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        x = normalize(torch.rand(size))
        return x


class BaseInspector():
    def __init__(self, name, module, store_in=False, store_out=False , device=torch.device('cpu')):
        self.device = device
        self.module = module
        self.name = name
        self.hooks = self._hook_layer()
        self.inputs = [] if store_in else None
        self.outputs = [] if store_out else None

    def _hook_layer(self):
        handle_o = self.module.register_forward_hook(self.hook_fn)
        handle_i = self.module.register_forward_pre_hook(self.pre_hook_fn)
        return handle_i, handle_o

    @staticmethod
    def _dump_fmap(layer_name,fmap,bw,scale,fformat="h",fname=None):
        fname = fname if (fname) else "{}_fmap_dump.txt".format(layer_name)
        print("{} output fmap will be dumped in {}".format(layer_name,fname))
        if fformat=="h":
            with open(fname,"w") as fout:
                fout.write("LAYER NAME:\t{}\n".format(layer_name))
                fout.write("\tSIZE: {}\tSHAPE: {}\t\n".format(fmap.size,fmap.shape))
                fout.write("\tSCALE: {}\tBITWIDTH: {}\t\n".format(scale,bw))
                fout.write("\n{}".format(fmap))

    def extract_param(self):
        raise NotImplementedError

    def dumpH_input_fmap(self,fname=None):
        if self.inputs:
            for n, i in enumerate(self.inputs):
                a = str(n) if len(self.inputs) > 1 else ""                    
                _i  = i.tensor.to('cpu').detach().numpy().copy()
                _is = i.scale.to('cpu').detach().numpy().copy()
                _ib = i.bit_width.to('cpu').detach().numpy().copy()
                self._dump_fmap(self.name, _i, _is, _ib, fname=fname+a)
        else:
            print("No hooked input tensor!")

    def dumpH_output_fmap(self,fname=None):
        if self.outputs:
            _o  = self.outputs.tensor.to('cpu').detach().numpy().copy()
            _os = self.outputs.scale.to('cpu').detach().numpy().copy()
            _ob = self.outputs.bit_width.to('cpu').detach().numpy().copy()
            self._dump_fmap(self.name, _o, _os, _ob, fname=fname)
        else:
            print("No hooked output tensor!")
            
    def pre_hook_fn(self, module, module_in):
        if self.inputs==[]:
            self.inputs.append(module_in[0].detach())
        else:
            pass
    
    def hook_fn(self, module, module_in, module_out):
        if self.outputs==[]:
            self.outputs=module_out.detach()
        else:
            pass

class ReLUInspector(BaseInspector):
    def __init__(self, name, module, store_in=False, store_out=False, device=torch.device('cpu')):
        super(ReLUInspector, self).__init__(name, module, store_in, store_out, device)

    def extract_param(self):
        return self.outputs.scale.to('cpu').detach().numpy().copy()

class BatchNormInspector(BaseInspector):
    def __init__(self, name, module, store_in=True, store_out=False, device=torch.device('cpu')):
        super(BatchNormInspector, self).__init__(name, module, True, store_out, device)
    
    def extract_param(self):
        if not self.inputs:
            print("Can't extract BN parameters. Please register input fmap before.")
            return -1
        else:
            w   = self.module.quant_weight().tensor.detach().numpy().copy()
            ws  = self.module.quant_weight_scale().detach().numpy().copy()
            iss = self.inputs[0].scale.to('cpu').detach().numpy().copy()
            ibw = self.outputs.bit_width
            b   = self.module.bias_quant(self.module.bias,  torch.tensor(iss*ws), ibw).tensor.detach().numpy().copy()
            bs  = iss*ws
            params = LayerParam(    weight = w,
                                    weight_scale = ws,
                                    bias = b,
                                    bias_scale = bs)
        return params


class PoolInspector(BaseInspector):
    def __init__(self, name, module, store_in=False, store_out=False, device=torch.device('cpu')):
        super(PoolInspector, self).__init__(name, module,store_in, store_out, device)

class ConvInspector(BaseInspector):
    def __init__(self, name, module, store_in=True, store_out=False, device=torch.device('cpu')):
        super(ConvInspector, self).__init__(name, module, store_in, store_out, device)
    
    def extract_param(self):
        w   = self.module.quant_weight().tensor.to('cpu').detach().numpy().copy()
        ws  = self.module.quant_weight_scale().to('cpu').detach().numpy().copy()
        b   = None
        bs  = None
        params = LayerParam(    weight = w,
                                weight_scale = ws,
                                bias = b,
                                bias_scale = bs)
        return params

class FCInspector(BaseInspector):
    def __init__(self, name, module, store_in=True, store_out=False, device=torch.device('cpu')):
        super(FCInspector, self).__init__(name, module, store_in, store_out, device)

class ResInspector(BaseInspector):
    def __init__(self, name, module, store_in=False, store_out=False, device=torch.device('cpu')):
        super(ResInspector, self).__init__(name, module,store_in, store_out, device)
    
    def pre_hook_fn(self, module, module_in):
        if self.inputs==[]:
            self.inputs.append(module_in[0].detach())
            self.inputs.append(module_in[1].detach())
        else:
            pass
    
    def extract_param(self):
        is0 = self.inputs[0].scale.to('cpu').detach().numpy().copy()
        is1 = self.inputs[1].scale.to('cpu').detach().numpy().copy()
        return is0,is1

ModuleClassDict = {
    QuantConv2d : ConvInspector,
    QuantReLU : ReLUInspector,
    BatchNorm2dToQuantScaleBias : BatchNormInspector,
    QuantAdaptiveAvgPool2d : PoolInspector,
    AimcConv2d : ConvInspector,
    QuantLinear : FCInspector,
    ResidualLayer : ResInspector,
    QuantIdentity : ReLUInspector}

class ModelInspector():
    def __init__(self, model, store_in, store_out, device=torch.device('cpu')):
        self.device = device
        self.model = model.to(device)
        self.inspectors = []
        self._hook_network(store_in,store_out)

    def _hook_network(self, st_in, st_out):
        for name, module in self.model.named_modules():
            if type(module) in ModuleClassDict.keys():
                self.inspectors.append(ModuleClassDict[type(module)](   name, 
                                                                        module,
                                                                        st_in,
                                                                        st_out,
                                                                        device))
    
    def _fw_p(self,image : ModelInputClass = None):
        if image==None:
            image = ModelInputClass()
        self.model.eval()
        self.model.to(self.device)
        return self.model(image.image)
    
    def dump_fmaps(self, path, image : ModelInputClass = None):
        _ = self._fw_p(image)
        for n, i in enumerate(self.inspectors):
            fname = "{}_fmap_dump.txt".format(i.name)
            i.dumpH_input_fmap(path+"{}_input_".format(n)+fname)
            i.dumpH_output_fmap(path+"{}_output_".format(n)+fname)