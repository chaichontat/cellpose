from cellpose import io, models, train
from subprocess import check_output, STDOUT
import os, shutil
import torch
import numpy as np
from pathlib import Path


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_class_train(data_dir):
    train_dir = str(data_dir.joinpath('2D').joinpath('train'))
    model_dir = str(data_dir.joinpath('2D').joinpath('train').joinpath('models'))
    shutil.rmtree(model_dir, ignore_errors=True)
    output = io.load_train_test_data(train_dir, mask_filter='_cyto_masks')
    images, labels, image_names, test_images, test_labels, image_names_test = output
    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu)
    cpmodel_path = train.train_seg(model.net, images, labels, train_files=image_names,
                                   test_data=test_images, test_labels=test_labels,
                                   test_files=image_names_test,
                                   save_path=train_dir, n_epochs=3)[0]
    io.add_model(cpmodel_path)
    io.remove_model(cpmodel_path, delete=True)
    print('>>>> model trained and saved to %s' % cpmodel_path)


def test_process_train_test_diameter_override_skips_inference(monkeypatch):
    calls = {"count": 0}

    def fake_diameters(lbl):
        calls["count"] += 1
        return 10.0, [1]

    monkeypatch.setattr(train.utils, "diameters", fake_diameters)

    img = np.zeros((3, 3), dtype=np.float32)
    lbl = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 0]], dtype=np.int32)

    out = train._process_train_test(
        train_data=[img],
        train_labels=[lbl],
        train_files=None,
        train_labels_files=None,
        train_probs=None,
        test_data=None,
        test_labels=None,
        test_files=None,
        test_labels_files=None,
        test_probs=None,
        load_files=True,
        min_train_masks=0,
        compute_flows=False,
        normalize_params={"normalize": False},
        channel_axis=None,
        device=torch.device("cpu"),
        diameter_override=60.0,
    )

    diam_train = out[5]
    assert np.allclose(diam_train, 60.0)
    assert calls["count"] == 0


def test_process_train_test_diameter_override_filters_empty_masks(monkeypatch):
    # Avoid heavy flow computation
    def fake_labels_to_flows(labels, **_kwargs):
        flows = []
        for lbl in labels:
            lbl = np.asarray(lbl)
            flows.append(
                np.stack(
                    (
                        lbl,
                        np.zeros_like(lbl),
                        np.zeros_like(lbl),
                        np.zeros_like(lbl),
                    ),
                    axis=0,
                )
            )
        return flows

    monkeypatch.setattr(train.dynamics, "labels_to_flows", fake_labels_to_flows)

    img = np.zeros((3, 3), dtype=np.float32)
    lbl_empty = np.zeros((3, 3), dtype=np.int32)
    lbl_full = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 0]], dtype=np.int32)

    out = train._process_train_test(
        train_data=[img, img],
        train_labels=[lbl_empty, lbl_full],
        train_files=None,
        train_labels_files=None,
        train_probs=None,
        test_data=None,
        test_labels=None,
        test_files=None,
        test_labels_files=None,
        test_probs=None,
        load_files=True,
        min_train_masks=1,
        compute_flows=False,
        normalize_params={"normalize": False},
        channel_axis=None,
        device=torch.device("cpu"),
        diameter_override=60.0,
    )

    train_data_out = out[0]
    diam_train = out[5]

    # The empty-mask sample should be filtered out when diameter_override is used
    assert len(train_data_out) == 1
    assert np.allclose(diam_train, 60.0)


def test_cli_train(data_dir):
    # import sys
    # path_root = Path(__file__).parents[1]
    # sys.path.append(str(path_root))
    # print(Path(__file__).parents[0],Path(__file__).parents[1],Path(__file__).parents[2])
    train_dir = str(data_dir.joinpath('2D').joinpath('train'))
    model_dir = str(data_dir.joinpath('2D').joinpath('train').joinpath('models'))
    shutil.rmtree(model_dir, ignore_errors=True)
    use_gpu = torch.cuda.is_available()
    gpu_str = "--use_gpu" if use_gpu else ""
    cmd = 'python -m cellpose %s --train --n_epochs 3 --dir %s --mask_filter _cyto_masks --pretrained_model None' % (gpu_str, train_dir)
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
    except Exception as e:
        print(e)
        raise ValueError(e)


def test_cli_make_train(data_dir):
    script_name = Path().resolve() / 'cellpose/gui/make_train.py'
    image_path = data_dir / '3D/gray_3D.tif'

    cmd = f'python {script_name} --image_path {image_path}'
    res = check_output(cmd, stderr=STDOUT, shell=True)

    # there should be 30 slices: 
    files = [f for f in (data_dir / '3D/train/').iterdir() if 'gray_3D' in f.name]
    assert 30 == len(files)

    shutil.rmtree((data_dir / '3D/train'))
