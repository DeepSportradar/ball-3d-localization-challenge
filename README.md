[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/JvMQgMkpkm)
[![Compete on EvalAI](https://badgen.net/badge/compete%20on/EvalAI/blue)](https://eval.ai/web/challenges/challenge-page/1685/overview)
[![Win $1,000.00](https://badgen.net/badge/win/%241%2C000.00/yellow)](http://mmsports.multimedia-computing.de/mmsports2022/challenge.html)
[![Kaggle Dataset](https://badgen.net/badge/kaggle/dataset/blue)](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset)

> **Note**: DeepSportRadar-**v2** challenges will come back next year (2023), stay tuned on our [Discord channel](https://discord.gg/JvMQgMkpkm)!


# DeepSportRadar Ball 3D localization challenge

One of the [ACM MMSports 2022 Workshop](http://mmsports.multimedia-computing.de/mmsports2022/index.html) challenges. An opportunity to publish or win a $1000,00 prize by competing on [EvalAI](https://eval.ai/web/challenges/challenge-page/1688/overview). See [this page](http://mmsports.multimedia-computing.de/mmsports2022/challenge.html) for more details.

<img src=assets/banner.png width=50%/>

**Table of contents**
- [Challenge rules](#challenge-rules)
- [Downloading the dataset](#downloading-the-dataset)
- [Dataset Splits](#dataset-splits)
- [Running the baseline](#running-the-baseline)
- [Participating with another codebase](#participating-with-another-codebase)
- [Metrics](#metrics)
- [Submissions](#submissions)
- [License](#license)

This challenge tackles the estimation of ball size on basketball scenes given oracle ball position. Using camera calibration information and knowledge of the real ball size, this estimation can be used to recover the ball 3d localization in the scene[^1].

## Challenge rules

Please refer to the challenge webpage for complete rules, timelines and awards: [https://deepsportradar.github.io/challenge.html](https://deepsportradar.github.io/challenge.html).

The goal of this challenge is to obtain the best estimation of ball size in pixels from true ball position given by an oracle. The metric used will be the mean absolute error (MAE) between the prediction and the ground-truth.
Contestants will be evaluated on the **challenge-set** for which labels will be kept secrets.

The competitors must conceive a model that relies only on the provided data for training. In the case of a neural-network based model, initial weights may come from a well-established public methods pret-trained on public data. This must be clearly stated in the publication/report.

## Downloading the dataset

The dataset can be downloaded [here](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset) and unzipped manually in the `basketball-instants-dataset/` folder of the project. To do it programmatically, you need the kaggle CLI:

```bash
pip install kaggle
```

Go to your Kaggle Account settings page and click on `Create new API Token` to download the file to be saved as `~/.kaggle/kaggle.json` for authentication. Finally download and unzip the dataset with:

```bash
kaggle datasets download deepsportradar/basketball-instants-dataset
unzip -qo ./basketball-instants-dataset.zip -d basketball-instants-dataset
```

The `basketball-instants-dataset` consists in raw images captured by the Keemotion system. For this challenge, we will only use thumbnails around the balls.


## Dataset splits

The challenge uses the split defined by [`DeepSportDatasetSplitter`](https://gitlab.com/deepsport/deepsport_utilities/-/blob/main/deepsport_utilities/ds/instants_dataset/dataset_splitters.py#L6) which
1. Uses images from `KS-FR-CAEN`, `KS-FR-LIMOGES` and `KS-FR-ROANNE` arenas for the **testing-set**.
2. Randomly samples 15% of the remaining images for the **validation-set**
3. Uses the remaining images for the **training-set**.

The **testing-set** should be used to evaluate your model, both on the **public** EvalAI leaderboard that provides the temporary ranking, and when communicating about your method.

The **challenge-set** should be used to submit the predictions of your model on the **private** EvalAI leaderboard that will be used for the official ranking. You are free to use the three sets defined above to build the final model on which your method will be evaluated, but the challenge set can only be used for computing the predictions of your model. 

The challenge set is available here: https://arena-data.keemotion.com/tmp/gva/ball_dataset_challenge.pickle.



## Running the baseline

The public https://github.com/gabriel-vanzandycke/deepsport repository provides a baseline for this challenge.
To use it, follow its installation instructions and add the folder `basketball-instants-dataset` full path to `DATA_PATH` in your `.env` file.

### Dataset pre-processing

The basline uses `ball_views.pickle`, a pre-processed dataset built from the full `basketball-instants-dataset` with the following script:
```bash
python deepsport/scripts/prepare_ball_views_dataset.py --dataset-folder basketball-instants-dataset
```

### Training the baseline

With the `ball_views.pickle` dataset ready and located in one of the paths of your `DATA_PATH`, you can train the baseline.
The configuration file `configs/ballsize.py` defines a model and the parameters to train it, as well as the necessary callbacks to compute the metrics. You can launch the model training by running:
```bash
python -m experimentator configs/ballsize.py --epochs 101 --kwargs "eval_epochs=range(0,101,20)"
```
Alternatively, you can use and adapt the provided notebook: [notebooks/run_ballsize_experiment.ipynb](https://github.com/gabriel-vanzandycke/deepsport/blob/main/notebooks/run_ballsize_experiment.ipynb).


### Run inference with the baseline

This repository provides a notebook to vizualize the baseline results and compute predictions on the testing and challenge sets: [load_baseline.ipynb](load_baseline.ipynb).



## Participating with another codebase

Participants are free to use their own codebase.
This repository offers a script to generate a dataset of input ball images and target ball size in pixel, with image side length given in argument. The official subsets can be generated with the `--subset` option:
```bash
python scripts/generate_dataset.pickle --dataset-folder basketball-instants-dataset --side-length 224 --subset trainval
python scripts/generate_dataset.pickle --dataset-folder basketball-instants-dataset --side-length 224 --subset test
```
The files created are [`mlworkflow.PickledDataset`](https://github.com/ispgroupucl/mlworkflow/blob/master/README.md)s of pairs (key, item) where items are dictionaries with:
- `"input_image"`: a `numpy.ndarray` RGB image thumbnail centered on the ball.
- `"ball_size"`: a `float` of the ball size in pixels.

This repository also implements a transformation that crops the dataset items with a given side length:
```python
from mlworkflow import PickledDataset, TransformedDataset
from tools.utils import CropCenterTransform
ds = PickledDataset("ball_dataset_trainval.pickle")
ds = TransformedDataset(ds, [CropCenterTransform(side_length=64)])
```

## Metrics

The goal of this challenge is to obtain the best estimation of ball size in pixels from true ball position given by an oracle. The metric used will be the mean absolute diameter error (MADE) between the prediction and the ground-truth. In addition, the mean absolute projection error (MAPE) and the mean absolute relative error (MARE), descsribed in[^1] are used for information.

## Submissions

The submission file can be generated using `tools.utils.PredictionsDumper` from this repository.
```python
with PredictionsDumper("predictions.json") as pd:
    for view_key in dataset.keys:
        prediction = compute(dataset.query_item(view_key))
        pd(view_key, float(prediction))
```

## Citation

If you use any DeepSportradar dataset in your research or wish to refer to the baseline results and discussion published in [our paper](https://arxiv.org/abs/2208.08190), please use the following BibTeX entry:

    @inproceedings{
        Van_Zandycke_2022,
        author = {Gabriel {Van Zandycke} and Vladimir Somers and Maxime Istasse and Carlo Del Don and Davide Zambrano},
	    title = {{DeepSportradar}-v1: Computer Vision Dataset for Sports Understanding with High Quality Annotations},
	    booktitle = {Proceedings of the 5th International {ACM} Workshop on Multimedia Content Analysis in Sports},
	    publisher = {{ACM}},
        year = 2022,
	    month = {oct},
        doi = {10.1145/3552437.3555699},
        url = {https://doi.org/10.1145%2F3552437.3555699}
    }
    

[^1]: [Ball 3D localization from a single calibrated image](https://arxiv.org/abs/2204.00003)
