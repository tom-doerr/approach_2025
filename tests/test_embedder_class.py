import numpy as np


def test_embedder_wraps_create_embedder(monkeypatch):
    import vkb.emb as emb

    def fake_create(name):
        def f(frame):
            return np.array([frame.shape[0], frame.shape[1]], dtype=np.float32)
        return f

    monkeypatch.setattr(emb, 'create_embedder', fake_create)
    E = emb.Embedder(model_name='fake')
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    v1 = E.embed(img)
    v2 = E(img)
    assert np.allclose(v1, np.array([10,20], dtype=np.float32))
    assert np.allclose(v2, v1)

