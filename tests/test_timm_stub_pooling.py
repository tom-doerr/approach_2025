import unittest, sys, types


class TestTimmStubAdaptivePool(unittest.TestCase):
    def test_stub_handles_224_and_64(self):
        import torch
        # Inject a tiny timm stub only for this test
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
        m = timm.create_model("stub-any", pretrained=False, num_classes=3)
        m.eval()
        for sz in (224, 64, 17):
            x = torch.zeros(2, 3, sz, sz); y = m(x)
            self.assertEqual(tuple(y.shape), (2, 3))
        # restore sys.modules
        if old_timm is not None: sys.modules['timm'] = old_timm
        else: del sys.modules['timm']


if __name__ == "__main__":
    unittest.main()
