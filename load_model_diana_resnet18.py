#%%
import torch
import numpy

from torchvision.datasets import ImageNet
from torchvision.transforms import transforms

from models.resnet.resnet18.model import qresnet18_factory    
from util.inspectors import ModelInspector, ModelInputClass
from util.compiler import ModelCompiler
from util.hw_model import SIMDModelClass

#%%

def check_output(hw_out_fmap,sw_out_fmap_f):
    for i in range(hw_out_fmap.shape[1]):
        s = numpy.sum(hw_out_fmap[0][i]-sw_out_fmap_f[0][i])
        if s!=0:
            for j in range(hw_out_fmap.shape[2]):
                ss = numpy.sum(hw_out_fmap[0][i][j]-sw_out_fmap_f[0][i][j])
                if ss!=0:
                    print(ss, i, j)
                    #print(hw_out_fmap[0][i][j])
                    #print(sw_out_fmap_f[0][i][j])
                    #exit()

if __name__ == '__main__':
    def _main():
        model = qresnet18_factory(
            model_type='aimc',
            config_file='/imec/other/nmorph/sarda74/diana/diana/diana/pytorch_models/zoo/models/resnet/resnet18/config_aimc.yml',
            trained_weights_file='/imec/other/nmorph/sarda74/diana/diana/diana/pytorch_models/zoo/models/resnet/resnet18/7b_2b_6b_aimc.pth'
        )
        #ld_support = ModelInspector(model, True, True)
        #ld_support.dump_fmaps(path="output/")
        #START USE EXAMPLE
        SIMD = SIMDModelClass()
        compiler = ModelCompiler(model)
        compiler.compile(ModelInputClass(path="imgnet_rand.pt"))
        #END USE EXAMPLE (following is debug)

        #compiler.compile(ModelInputClass([1,3,224,224]))
        bn_layer = 24
        if bn_layer==3:
            in_fmap_raw = compiler.inspector.inspectors[bn_layer].inputs[0].tensor.to('cpu').detach().numpy().copy()
            in_fmap = in_fmap_raw/compiler.inspector.inspectors[bn_layer].inputs[0].scale.to('cpu').detach().numpy().copy()
            sw_out_fmap_f_raw = compiler.inspector.inspectors[bn_layer+1].outputs.tensor.to('cpu').detach().numpy().copy()
            sw_out_fmap_f = sw_out_fmap_f_raw/compiler.inspector.inspectors[bn_layer+1].outputs.scale.to('cpu').detach().numpy().copy()
            hw_out_fmap = SIMD.fp(compiler.analogConvLayers[0],in_fmap)
            check_output(hw_out_fmap,sw_out_fmap_f)
        if bn_layer==6:
            in_fmap_raw = compiler.inspector.inspectors[bn_layer].inputs[0].tensor.to('cpu').detach().numpy().copy()
            in_fmap = in_fmap_raw/compiler.inspector.inspectors[bn_layer].inputs[0].scale.to('cpu').detach().numpy().copy()
            res_fmap_raw = compiler.inspector.inspectors[bn_layer+2].inputs[1].tensor.to('cpu').detach().numpy().copy()
            res_fmap = res_fmap_raw/compiler.inspector.inspectors[bn_layer+2].inputs[1].scale.to('cpu').detach().numpy().copy()
            sw_out_fmap_f_raw = compiler.inspector.inspectors[bn_layer+1].outputs.tensor.to('cpu').detach().numpy().copy()
            sw_out_fmap_f = sw_out_fmap_f_raw/compiler.inspector.inspectors[bn_layer+1].outputs.scale.to('cpu').detach().numpy().copy()
            hw_out_fmap = SIMD.fp(compiler.analogConvLayers[1],in_fmap,res_fmap)
            check_output(hw_out_fmap,sw_out_fmap_f)
        if bn_layer==24:
            in_fmap_raw = compiler.inspector.inspectors[bn_layer].inputs[0].tensor.to('cpu').detach().numpy().copy()
            in_fmap = in_fmap_raw/compiler.inspector.inspectors[bn_layer].inputs[0].scale.to('cpu').detach().numpy().copy()

            out_fmap_bn_raw = compiler.inspector.inspectors[bn_layer].outputs.tensor.to('cpu').detach().numpy().copy()
            out_fmap_bn = out_fmap_bn_raw/compiler.inspector.inspectors[bn_layer].outputs.scale.to('cpu').detach().numpy().copy()

            downsample_in_fmap_raw = compiler.inspector.inspectors[bn_layer+3].inputs[0].tensor.to('cpu').detach().numpy().copy()
            downsample_in_fmap = downsample_in_fmap_raw/compiler.inspector.inspectors[bn_layer+2].inputs[0].scale.to('cpu').detach().numpy().copy()

            downsample_out_fmap_raw = compiler.inspector.inspectors[bn_layer+3].outputs.tensor.to('cpu').detach().numpy().copy()
            downsample_out_fmap = downsample_out_fmap_raw/compiler.inspector.inspectors[bn_layer+3].outputs.scale.to('cpu').detach().numpy().copy()
            
            sw_res_fmap_out_raw = compiler.inspector.inspectors[bn_layer+4].outputs.tensor.to('cpu').detach().numpy().copy()
            sw_res_fmap_out = sw_res_fmap_out_raw/compiler.inspector.inspectors[bn_layer+4].outputs.scale.to('cpu').detach().numpy().copy()

            sw_out_fmap_f_raw = compiler.inspector.inspectors[bn_layer+1].outputs.tensor.to('cpu').detach().numpy().copy()
            sw_out_fmap_f = sw_out_fmap_f_raw/compiler.inspector.inspectors[bn_layer+1].outputs.scale.to('cpu').detach().numpy().copy()
            
            res_fmap = SIMD.fp_dbg(compiler.analogConvLayers[6],downsample_in_fmap,ds=True)
            hw_out_fmap_sw = SIMD.fp_dbg(compiler.analogConvLayers[5],in_fmap,numpy.floor(downsample_out_fmap/4))
            hw_out_fmap = SIMD.fp_dbg(compiler.analogConvLayers[5],in_fmap,res_fmap[-1])

            check_output(hw_out_fmap[0],out_fmap_bn)
            check_output(res_fmap[0],downsample_out_fmap)           
            #check_output(res_fmap[-1],numpy.floor(downsample_out_fmap/4))
            check_output(hw_out_fmap[-1], sw_out_fmap_f)

    _main()

# %%
