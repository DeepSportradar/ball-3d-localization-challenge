# Ball 3D localization


## Task

The task consists in finding ball 3D localization from a single calibrated image. The most natural approach is to detect the ball and estimate its diameter in the image space, and use calibration data and knowledge of real ball diameter in meters to compute the localization in the 3D space.
This approach is described in (this paper)[!url]

## Challenge
Given a dataset of ball thumbnails and ground-truth ball diameter in pixels, you are asked to create a model that predicts ball diameter on unseen images of balls.

## Metric
The metric used is the Mean Absolute Error in pixels

## Baseline
