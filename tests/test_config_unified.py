import os, unittest
from types import SimpleNamespace


class TestUnifiedConfig(unittest.TestCase):
    def test_env_overrides_visdom(self):
        from vkb.config import make_config
        os.environ['VKB_VISDOM_ENV'] = 'myenv'
        os.environ['VKB_VISDOM_PORT'] = '8123'
        args = SimpleNamespace(data='data', embed_model='m', clf='dl', epochs=1, batch_size=4, lr=1e-4, wd=0.0, device='cpu', visdom_env='ignored', visdom_port=0)
        cfg = make_config(args)
        assert cfg.visdom_env == 'myenv'
        assert cfg.visdom_port == 8123


if __name__ == '__main__':
    unittest.main()

