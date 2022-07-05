import argparse
from importlib_resources import files
import json
import os

from tqdm.auto import tqdm

from mlworkflow import PickledDataset, TransformedDataset, FilteredDataset
from deepsport_utilities import import_dataset, InstantsDataset, deserialize_keys
from deepsport_utilities.transforms import DataExtractorTransform
from deepsport_utilities.ds.instants_dataset import BuildBallViews, ViewsDataset, DownloadFlags
from deepsport_utilities.ds.instants_dataset.views_transforms import AddBallAnnotation, AddBallSizeFactory, AddImageFactory
from deepsport_utilities.ds.instants_dataset.dataset_splitters import DeepSportDatasetSplitter


parser = argparse.ArgumentParser(description="""
    This script enables participants to create their own dataset for the challenge.
    The dataset is saved as an `mlworkflow.PickledDataset` of pairs (view_key, item)
    where view_key is the dataset key (a tuple) and item is a dictionary with fields:
        - 'image': a `numpy.ndarray` RGB image thumbnail centered on the ball. The
            thumbnail size is given in arguments.
        - 'size': a `float` of the ball size in pixels.
    Only balls flagged as visible are kept.
""")
parser.add_argument("--dataset-folder", required=True, help="Basketball Instants Dataset folder")
parser.add_argument("--output-folder", default=None, help="Folder in which specific dataset will be created. Defaults to `dataset_folder` given in arguments.")
parser.add_argument("--side-length", required=True, type=int, help="Ball thumbnail side length (integer)")
parser.add_argument("--subset", default='trainvaltest', choices=['train', 'val', 'test', 'trainval', 'trainvaltest', 'challenge'], help="Dataset split")
args = parser.parse_args()

# The `dataset_config` is used to create each dataset item
dataset_config = {
    "download_flags": DownloadFlags.WITH_IMAGE | DownloadFlags.WITH_CALIB_FILE,
    "dataset_folder": args.dataset_folder  # informs dataset items of raw files location
}

# Import dataset
filename = "basketball-instants-dataset.json" if args.subset != 'challenge' else "mmsports-instants-dataset-challenge-set_dataset.json"
database_file = os.path.join(args.dataset_folder, filename)
ds = import_dataset(InstantsDataset, database_file, **dataset_config)


# Extract subset keys
if args.subset != 'challenge':
    keys = []
    for subset_name, subset_keys in zip(['train', 'val', 'test'], DeepSportDatasetSplitter().split_keys(ds.keys.all())):
        if subset_name in args.subset:
            keys += subset_keys
    ds = FilteredDataset(ds, predicate=lambda k,v: k in keys)


# Build a dataset of balls centered in the image with a margin of 'side_length' pixels around the ball
ds = ViewsDataset(ds, BuildBallViews(margin=args.side_length, margin_in_pixels=True, padding=args.side_length))


# Transform Views objects into dictionaries with only image and ball size in pixels
ds = TransformedDataset(ds, [
    DataExtractorTransform(    # Transforms python object in dictionary
        AddImageFactory(),     #    Adds image to the dictionary
        AddBallSizeFactory(),  #    Adds ball size to the dictionary
    )
])

# Ignore balls flagged as not visible
ds = FilteredDataset(ds, lambda k,v: v['ball_size'] != np.nan)

# Write dataset as an mlworkflow.PickledDataset
output_folder = args.output_folder or args.dataset_folder
path = os.path.join(output_folder, f"ball_dataset_{args.subset}.pickle")
PickledDataset.create(ds, path, yield_keys_wrapper=tqdm)
print(f"Successfully generated {path}")

