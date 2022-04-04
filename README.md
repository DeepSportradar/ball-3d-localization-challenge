# CHALLENGE STARTS OFFICIALLY APRIL 6th. Stayed tuned.


[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/JvMQgMkpkm) [![Compete on EvalAI](https://badgen.net/badge/compete%20on/EvalAI/blue)](https://eval.ai/web/challenges/challenge-page/1685/overview) [![Win $1,000.00](https://badgen.net/badge/win/%241%2C000.00/yellow)](http://mmsports.multimedia-computing.de/mmsports2022/challenge.html) [![Kaggle Dataset](https://badgen.net/badge/kaggle/dataset/blue)](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset)


# DeepSportRadar Ball 3D localization challenge

One of the [ACM MMSports 2022 Workshop](http://mmsports.multimedia-computing.de/mmsports2022/index.html) challenges. An opportunity to publish, as well as a $1000,00 prize. See [this page](http://mmsports.multimedia-computing.de/mmsports2022/challenge.html) for more details.

**Table of contents**
- [Challenge rules](#challenge-rules)
- [Installation](#installation)
  - [Downloading the dataset](#downloading-the-dataset)
  - [The mlworkflow dataset format](#mlworkflow-dataset-format)
  - [About the splits](#about-the-splits)
- [Using deepsport repository](#using-deepsport)
  - [Installation](#installation-1)
  - [Baseline](#baseline)
  - [Test, metrics and submission](#test-metrics-and-submission)
- [Participating with another codebase](#participating-with-another-codebase)
  - [Submission format](#submission-format)
  - [Computing metrics](#computing-metrics)
- [License](#license)

This challenge tackles the estimation of ball size on basketball scenes. Using camera calibration information and knowledge of the real ball size, this estimation can be used to recover the ball 3d localization in the scene[^1].

## Challenge rules

The goal is to obtain the best estimation of ball size in pixels on an evaluation set -- called *challenge*-set -- for which labels will be kept secrets. The metric used will be the mean absolute error (MAE) between the prediction and the secret ground-truth. The *challenge*-set will be provided later.

The competitors must conceive a model that relies only on the provided data for training. In the case of a neural-network based model, initial weights may come from a well-established public methods pret-trained on public data. **This must be clearly stated in the publication/erport**.

Please see the challenge page for more details: [https://deepsportradar.github.io/challenge.html](https://deepsportradar.github.io/challenge.html).

## Installation

### Downloading the dataset

The dataset can be found [here](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset). It can be downloaded and unzipped manually in the `deepsport_dataset/` folder of the project.

We will here download it programmatically. First install the kaggle CLI:

```bash
pip install kaggle
```

Go to your Kaggle Account settings page and click on `Create new API Token` to download the file to be saved as `~/.kaggle/kaggle.json` for authentication. Finally download and unzip the dataset using kaggle CLI:

```bash
kaggle datasets download deepsportradar/basketball-instants-dataset
unzip -qo ./basketball-instants-dataset.zip -d deepsport_dataset
```


## Challenge
Given a dataset of ball thumbnails and ground-truth ball diameter in pixels, you are asked to create a model that predicts ball diameter on unseen images of balls.

## Baseline


## Competition awards


## timeline

[^1]: [Ball 3D localization from a single calibrated image](!https://arxiv.org/abs/2204.00003)
