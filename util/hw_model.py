import numpy
from util.compiler import ConvLayerClass

class SIMDModelClass():
    def _bn(self, layer, in_fmap):
        out_fmap = numpy.zeros(in_fmap.shape)
        for c in range(in_fmap.shape[1]):
            out_fmap[0][c] = (layer.bn_w[c][0]*(2**layer.bn_ws))*in_fmap[0][c] + (layer.bn_b[c]*2**layer.bn_bs)
        return out_fmap

    def _rs(self, layer, in_fmap0, in_fmap1):
        out_fmap = in_fmap0 + in_fmap1*(2**layer.r_s)
        return out_fmap
    
    def _relu(self, in_fmap):
        out_fmap = in_fmap*(in_fmap>0).astype(int)
        return out_fmap

    def _clip(self, in_fmap, bw):
        max_val = 2**(bw-1)-1
        min_val = -2**(bw-1)+1
        clip_min_mask = (in_fmap<min_val).astype(int)
        clip_max_mask = (in_fmap>max_val).astype(int)
        clip_min_mask_n = numpy.logical_xor(clip_min_mask, 1).astype(int)
        clip_max_mask_n = numpy.logical_xor(clip_max_mask, 1).astype(int)
        min_filler = numpy.ones(in_fmap.shape)*clip_min_mask*min_val
        max_filler = numpy.ones(in_fmap.shape)*clip_max_mask*max_val
        out_fmap = (in_fmap*clip_min_mask_n*clip_max_mask_n)+min_filler+max_filler
        return out_fmap

    def _quant(self, layer, in_fmap, ds=False):
        out_fmap = numpy.floor(in_fmap/2**layer.q_s)
        if ds:
            out_fmap = self._clip(out_fmap,7)
        return out_fmap

    def fp(self, layer, in_fmap0, in_fmap1=None, ds=False):
        x = self._bn(layer, in_fmap0)
        if type(in_fmap1) == numpy.ndarray:
            x = self._rs(layer, x, in_fmap1)
        if ds==False:
            x = self._relu(x)
        x = self._quant(layer, x)
        return x

    def fp_dbg(self, layer, in_fmap0, in_fmap1=None, ds=False):
        out = []
        out.append(self._bn(layer, in_fmap0))
        i=0
        if type(in_fmap1) == numpy.ndarray:
            out.append(self._rs(layer, out[i], in_fmap1))
            i+=1
        if ds==False:
            out.append(self._relu(out[i]))
            i+=1
        out.append(self._quant(layer, out[i]))
        return out