import os, tempfile


def test_io_class_paths_and_listing():
    from vkb.io import IO
    with tempfile.TemporaryDirectory() as tmp:
        io = IO(data_root=os.path.join(tmp, 'data'))
        p = io.make_output_path('A')
        # path directory is data/A and endswith .mp4
        assert os.path.dirname(p).endswith(os.path.join('data','A'))
        assert p.endswith('.mp4')
        # create a couple of videos to list
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'wb').close()
        p2 = io.make_output_path('A')
        open(p2, 'wb').close()
        vids = io.list_videos()
        # returns list of (path,label) pairs
        assert all(isinstance(v, tuple) and len(v)==2 for v in vids)
        assert all(v[1]=='A' for v in vids)


def test_io_update_symlink():
    from vkb.io import IO
    with tempfile.TemporaryDirectory() as tmp:
        io = IO(data_root=os.path.join(tmp, 'data'))
        p = io.make_output_path('B')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'wb').close()
        link = io.update_latest_symlink(p)
        assert os.path.islink(link)
        # link target is basename
        assert os.readlink(link) == os.path.basename(p)
