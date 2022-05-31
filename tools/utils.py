from deepsport_utilities.transforms import Transform
from deepsport_utilities.ds.instants_dataset.views_dataset import ViewKey, View


class CropBallViewTransform(Transform):
    """ Transforms views centered on ball by cropping a square thumbnail of the ball.
        Arguments:
            - side_length: thumbnail side length in pixels
    """
    def __init__(self, side_length: int):
        self.side_length = side_length
    def __call__(self, view_key: ViewKey, view: View):
        w, h, _ = view.image.shape
        sl = self.side_length
        x_slice = slice((w-sl)//2, (w-sl)//2 + sl, None)
        y_slice = slice((h-sl)//2, (h-sl)//2 + sl, None)
        return {"input_image": view.image[y_slice, x_slice]}

class PredictionsDumper():
    def __init__(self, filename):
        self.predictions = []
        self.filename = filename
    def __enter__(self):
        pass
    def __call__(self, view_key, prediction: float):
        self.predictions.append({
            "arena_label": view_key.arena_label,
            "game_id": view_key.game_id,
            "timestamp": view_key.timestamp,
            "camera": view_key.camera,
            "index": view_key.index,
            "prediction": prediction
        })
    def __exit__(self):
        json.dump(self.predictions, open(filename, "w"))
        print(f"{filename} successfully written")