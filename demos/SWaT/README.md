### System identification from time sereis for SWaT dataset
---
[Secure Water Treatment (SWaT) Dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) is one of the free available dataset containing cyberattacks on industrial control systems. A cyber attack protection system can be implemented based on a predictive model for signals. If the prediction error is greater than a statistically matched threshold defined at the training stage, an anomaly is detected as a potential cyberattack.

In this example, a simple predictive model for a small subset of SWaT dataset is built. Model fitting is implemented using time series nested cross-validation.