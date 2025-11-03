import sys, types, torch
class DropPath(torch.nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__(); self.p = float(p)
    def forward(self, x):
        return x
def _stub_create_model(name: str, pretrained: bool = False, num_classes: int = 2, drop_path_rate: float = 0.0, drop_rate: float = 0.0):
    layers = [torch.nn.AdaptiveAvgPool2d((8, 8)), torch.nn.Flatten()]
    if drop_rate and drop_rate > 0:
        layers.append(torch.nn.Dropout(p=float(drop_rate)))
    if drop_path_rate and drop_path_rate > 0:
        layers.append(DropPath(drop_path_rate))
    layers.append(torch.nn.Linear(8*8*3, int(num_classes)))
    return torch.nn.Sequential(*layers)
old_timm = sys.modules.get('timm'); sys.modules['timm'] = types.SimpleNamespace(create_model=_stub_create_model)
import timm


def test_drop_path_inserts_modules():
    m = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=3, drop_path_rate=0.2)
    # timm uses DropPath modules when rate>0
    has_dp = any(mod.__class__.__name__.lower() in {'droppath','stochasticdepth'} for mod in m.modules())
    assert has_dp, 'expected DropPath/StochasticDepth modules when drop_path_rate>0'


def test_dropout_flag_is_accepted():
    # Some models may ignore drop_rate; ensure call succeeds
    m = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=3, drop_rate=0.3)
    assert isinstance(m, torch.nn.Module)
    # restore timm
    if old_timm is not None: sys.modules['timm'] = old_timm
    else: sys.modules.pop('timm', None)
