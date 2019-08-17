### System identification from time series for SWaT dataset
---
[Secure Water Treatment (SWaT) Dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) is one of the free available dataset containing cyberattacks on industrial control systems. A cyber attack protection system can be implemented based on a predictive model for signals. If the prediction error is greater than a statistically matched threshold defined at the training stage, an anomaly is detected as a potential cyberattack.

In this example, a simple predictive model for a small subset of SWaT dataset is built. Model fitting is implemented using time series nested cross-validation.

### Data set
In this example, we use only a small subset of SWaT dataset (SWaT_P1.csv). We examine only three signals:
**LIT101** is a state of the system (water tank level),
**MV101** controls raw water tank, and
**P101** is control that pumps water from raw water tank to next stage P2.

| Timestamp  | LIT101 | MV101 | P101 |
| :------------ |:---------------:| :-----:| :-----:|
| ... | ... | ... | ... |
| 28/12/2015 10:12:23 AM | 535.5254 | 2 | 2 |
| 28/12/2015 10:12:24 AM | 536.2319 | 2 | 1 |
| ... | ... | ... | ... |


### Equation-based dynamics description
The system dynamics can be described in general form

$$\frac{d}{dt}X(t)=F(X(\tau), u, v)$$

where *X* for **LIT101**, *u* for control **MV101**, and *v* for control **P101**. Note that the right-hand side of the equation can be time delayed.


### Data-driven model

As a discrete data-driven approximation of the unknown differential equation, a polynomial regression can be used

$$X_i=LinearRegression(PolynomialFeatures(X_i, X_{i-1}, \ldots, X_{i-delay}, u_i, u_{i-1}, \ldots, u_{i-delay}, v_i, v_{i-1}, \ldots, v_{i-delay},))$$.

An implementation of this model and nested cross-validation performance estimation is presented in this demo.