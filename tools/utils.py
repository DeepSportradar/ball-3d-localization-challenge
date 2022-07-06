import json

from deepsport_utilities.transforms import Transform
from deepsport_utilities.ds.instants_dataset.views_dataset import ViewKey, View


class CropCenterTransform(Transform):
    """ Crops an image from its center with a given side length
        Arguments:
            - side_length: crop side length in pixels
    """
    def __init__(self, side_length: int):
        self.side_length = side_length
    def __call__(self, view_key, item):
        w, h, _ = item['input_image'].shape
        sl = self.side_length
        x_slice = slice((w-sl)//2, (w-sl)//2 + sl, None)
        y_slice = slice((h-sl)//2, (h-sl)//2 + sl, None)
        item['input_image'] = item['input_image'][y_slice, x_slice]
        return item

class PredictionsDumper():
    def __init__(self, filename):
        self.predictions = []
        self.filename = filename
    def __enter__(self):
        return self
    def __call__(self, view_key, prediction: float):
        self.predictions.append({
            "arena_label": view_key.arena_label,
            "game_id": view_key.game_id,
            "timestamp": view_key.timestamp,
            "camera": view_key.camera,
            "index": view_key.index,
            "prediction": prediction
        })
    def __exit__(self ,type, value, traceback):
        json.dump(self.predictions, open(self.filename, "w"))
        print(f"{self.filename} successfully written")