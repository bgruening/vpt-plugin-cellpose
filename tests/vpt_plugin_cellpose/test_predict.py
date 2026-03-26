from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from vpt_core.io.image import ImageSet
from vpt_plugin_cellpose import CellposeSegParameters, CellposeSegProperties
from vpt_plugin_cellpose import predict


@dataclass(frozen=True)
class Circle:
    x: int
    y: int
    radius: int


def generate_images(image_size: int, cells: List[Circle]) -> Tuple[ImageSet, str, str]:
    dapi = np.ones((image_size, image_size), dtype=np.uint16)
    polyt = np.ones((image_size, image_size), dtype=np.uint16)
    for cell in cells:
        cv2.circle(dapi, (cell.x, cell.y), int(cell.radius * 0.8), (255, 255, 255), -1)
        cv2.circle(polyt, (cell.x, cell.y), int(cell.radius * 1.5), (255, 255, 255), -1)

    nuclear_channel, fill_channel = "DAPI", "PolyT"
    images = ImageSet()
    images[nuclear_channel] = {i: dapi for i in range(3)}
    images[fill_channel] = {i: polyt for i in range(3)}
    return images, nuclear_channel, fill_channel


class FakeCellposeModel:
    init_kwargs = []
    eval_kwargs = []

    def __init__(self, gpu: bool = False, pretrained_model: str | None = None, model_type: str | None = None) -> None:
        self.gpu = gpu
        self.pretrained_model = pretrained_model
        self.model_type = model_type
        FakeCellposeModel.init_kwargs.append(
            {
                "gpu": gpu,
                "pretrained_model": pretrained_model,
                "model_type": model_type,
            }
        )

    def eval(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        FakeCellposeModel.eval_kwargs.append(kwargs)
        mask = np.zeros(image.shape[:3], dtype=np.int32)
        height, width = mask.shape[1], mask.shape[2]

        labels = [
            (slice(10, min(30, height)), slice(10, min(30, width))),
            (slice(45, min(70, height)), slice(45, min(70, width))),
            (slice(90, min(120, height)), slice(20, min(50, width))),
            (slice(140, min(170, height)), slice(140, min(170, width))),
        ]
        for label_index, (row_slice, col_slice) in enumerate(labels, start=1):
            mask[:, row_slice, col_slice] = label_index

        return (mask.reshape(-1),)


@pytest.fixture(autouse=True)
def reset_fake_model() -> None:
    FakeCellposeModel.init_kwargs.clear()
    FakeCellposeModel.eval_kwargs.clear()


def test_run_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    cells = [Circle(20, 15, 10), Circle(30, 100, 10), Circle(100, 20, 15), Circle(210, 100, 15)]
    images, nuc, fill = generate_images(256, cells)
    monkeypatch.setattr(predict.models, "CellposeModel", FakeCellposeModel)

    properties = CellposeSegProperties("cyto2", "2D", "latest", None)
    parameters = CellposeSegParameters(nuc, fill, 30, 0.95, -5.5, 256)
    mask = predict.run(images, properties, parameters)

    assert mask.shape == (3, 256, 256)
    for i in images.z_levels():
        labels = np.unique(mask[i, :, :])
        assert labels.tolist() == [0, 1, 2, 3, 4]

    assert FakeCellposeModel.init_kwargs == [{"gpu": False, "pretrained_model": None, "model_type": "cyto2"}]
    assert FakeCellposeModel.eval_kwargs == [
        {
            "z_axis": None,
            "channel_axis": None,
            "diameter": 30,
            "flow_threshold": 0.95,
            "cellprob_threshold": -5.5,
            "resample": False,
            "min_size": 256,
            "do_3D": False,
        }
    ]
