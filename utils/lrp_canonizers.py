import torch
from torchvision.models.efficientnet import MBConv
from torchvision.ops.misc import SqueezeExcitation
from zennit import canonizers as canonizers
from zennit import layer as zlayer


class SignalOnlyGate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2):
        return x1 * x2

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output), grad_output


class SECanonizer(canonizers.AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, SqueezeExcitation):
            attributes = {
                'forward': cls.forward.__get__(module),
                'fn_gate': SignalOnlyGate()
            }
            return attributes
        return None

    @staticmethod
    def forward(self, input):
        scale = self._scale(input)
        return self.fn_gate.apply(scale, input)


class MBConvCanonizer(canonizers.AttributeCanonizer):
    '''Canonizer specifically for MBConvBlock of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, MBConv):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': zlayer.Sum()
            }
            return attributes
        return None

    @staticmethod
    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)

            # result += input
            result = torch.stack([input, result], dim=-1)
            result = self.canonizer_sum(result)
        return result


class EfficientNetBNCanonizer(canonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            SECanonizer(),
            MBConvCanonizer(),
            canonizers.SequentialMergeBatchNorm()
        ))
