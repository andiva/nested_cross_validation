<p align="center">
  <img src="https://raw.githubusercontent.com/Pauleech/kaia-machine-learning-home-task/master/images/nested.png" width="400" />
</p>

---
**Time Series Nested Cross-Validation** replaces traditional cross-validation for time series data sets. The repository contains of implementation of generator for (train, test) indices.

### License
MIT License. Anyone can freely use.

### Install

Clone or dowload source code to your local directory and run 
```sh
user@local:/src_directory$ pip install .
```
To uninstall package run
```sh
$ pip uninstall nested_cross_validation
```
## Example 1: indices of train\test folds
```python
from nested_cross_validation.loops import uniform_loop

num_samples = 100
num_folds= 5
split_ratio = 4

fold_indices = uniform_loop(0, num_samples-1, num_folds, split_ratio

for trn, tst in fold_indices:
    print(trn, tst)
```
Expected result:
```sh
train:  [0, 44] test:  [45, 55]
train:  [11, 55] test:  [56, 66]
train:  [22, 66] test:  [67, 77]
train:  [33, 77] test:  [78, 88]
train:  [44, 88] test:  [89, 99]
```

## Example 2: indices of outer and inner loops
```python
import  nested_cross_validation as ncv

num_samples = 100
num_folds_outer = 5
split_ratio_outer = 4
num_folds_inner = 2
split_ratio_inner = 1

fold_indices = get_fold_indices(num_samples, num_folds_outer, split_ratio_outer,
                                             num_folds_inner, split_ratio_inner)

for i in range(num_folds_outer):
    outer_id = str(i).zfill(2)
    trn, tst = next(fold_indices)
    print(f"trn-outer-{outer_id}", trn)
    print(f"tst-outer-{outer_id}", tst)
    for j in range(num_folds_inner):
        inner_id = str(j).zfill(2)
        trn, tst = next(fold_indices)
        print(f"trn-inner-{outer_id}-{inner_id}", trn)
        print(f"tst-inner-{outer_id}-{inner_id}", tst)
```

```sh
trn-outer-00 [0, 44]
tst-outer-00 [45, 55]
trn-inner-00-00 [0, 14]
tst-inner-00-00 [15, 29]
trn-inner-00-01 [15, 29]
tst-inner-00-01 [30, 44]
trn-outer-01 [11, 55]
tst-outer-01 [56, 66]
trn-inner-01-00 [11, 25]
tst-inner-01-00 [26, 40]
trn-inner-01-01 [26, 40]
tst-inner-01-01 [41, 55]
trn-outer-02 [22, 66]
tst-outer-02 [67, 77]
trn-inner-02-00 [22, 36]
tst-inner-02-00 [37, 51]
trn-inner-02-01 [37, 51]
tst-inner-02-01 [52, 66]
trn-outer-03 [33, 77]
tst-outer-03 [78, 88]
trn-inner-03-00 [33, 47]
tst-inner-03-00 [48, 62]
trn-inner-03-01 [48, 62]
tst-inner-03-01 [63, 77]
trn-outer-04 [44, 88]
tst-outer-04 [89, 99]
trn-inner-04-00 [44, 58]
tst-inner-04-00 [59, 73]
trn-inner-04-01 [59, 73]
tst-inner-04-01 [74, 88]
```