[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/JvMQgMkpkm)
[![Compete on EvalAI](https://badgen.net/badge/compete%20on/EvalAI/blue)](https://eval.ai/web/challenges/challenge-page/1685/overview)
[![Win $1,000.00](https://badgen.net/badge/win/%241%2C000.00/yellow)](http://mmsports.multimedia-computing.de/mmsports2022/challenge.html)
[![Kaggle Dataset](https://badgen.net/badge/kaggle/dataset/blue)](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset)


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

The **testing-set** should be used to evaluate your model, both on the public EvalAI leaderboard that provides the temporary ranking, and when communicating about your method.

The **challenge-set** will be used for the official ranking, and you are free to use the three sets defined above to build the final model on which your method will be evaluated in the EvalAI submission. The challenge set is available here: https://arena-data.keemotion.com/tmp/gva/mmsports_challenge_set_data.zip. Images in which the ball is not visible are flagged as such in our ground-truth and won't be used to compute the metric.



## Running the baseline

The public https://github.com/gabriel-vanzandycke/deepsport repository provides a baseline for this challenge.
To use it, follow its installation instructions and add the folder `basketball-instants-dataset` full path to `DATA_PATH` in your `.env` file.

### Dataset pre-processing

The basline uses a pre-processed dataset built from the `basketball-instants-dataset` with the following script:
```bash
python deepsport/scripts/prepare_ball_views_dataset.py --dataset-folder basketball-instants-dataset
```

### Training the baseline

With the dataset ready, you can train the baseline.
The configuration file `configs/ballsize.py` defines a model and the parameters to train it, as well as the necessary callbacks to compute the metrics. You can launch the model training by running:
```bash
python -m experimentator configs/ballsize.py --epochs 101 --kwargs "eval_epochs=range(0,101,20)"
```

You can vizualize the training process the following way:

```python
import dotenv
dotenv.load_dotenv()
import numpy as np
from matplotlib import pyplot as plt
from experimentator import DataCollector
dc = DataCollector("{os.environ['RESULTS_FOLDER']}/ballsize/latest/history.dcp")
fig, axes = plt.subplots(2,1)
for ax, metric in zip(axes, ["loss", "MADE"]):
    for subset in ["training", "validation"]:
        label = f"{subset}_{metric}"
        l = np.array(dc[label, :])
        w = np.where(l)[0]
        ax.plot(w, l[w], label=label)
    ax.legend()
```

### Inferrence

You can run the trained model the following way:
```python
import os
from matplotlib import pyplot as plt
from tasks.ballsize import CropBallTransform
from experimentator import build_experiment, find, collate_fn
from mlworkflow import PickledDataset, TransformedDataset

exp = build_experiment(os.path.join(os.environ['RESULTS_FOLDER'], "ballsize/latest/config.py"), robust=True)
ds = PickledDataset(find("ball_views.pickle"))
ds = TransformedDataset(ds, [CropBallTransform(exp.cfg["side_length"])])

for keys, data in ds.batches(batch_size=1, collate_fn=collate_fn):
    output = exp.predict(data)
    plt.imshow(data["batch_input_image"][0])
    plt.title("{:.2f}".format(output["predicted_diameter"][0]))
    break
```

## Participating with another codebase

Participants are free to use their own codebase.
This repository offers a script to generate a dataset of input ball images and target ball size in pixel, with image side length given in argument. Additionally, the official subsets can be generated with the `--subset` option:
```bash
python tools/generate_dataset.pickle --dataset-folder basketball-instants-dataset --side-length 64 --subset trainval
```
The file created is an [`mlworkflow.PickledDataset`](https://github.com/ispgroupucl/mlworkflow/blob/master/README.md) of pairs (key, item) where keys are item identifiers and items are a dictionaries with:
- `"image"`: a `numpy.ndarray` RGB image thumbnail centered on the ball.
- `"size"`: a `float` of the ball size in pixels.

If `--subset challenge` option is given, the challenge evaluation in which ball size is always `numpy.nan` will be used.

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



[^1]: [Ball 3D localization from a single calibrated image](https://arxiv.org/abs/2204.00003)
