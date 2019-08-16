import warnings

def uniform_loop(start, end, num_folds, split_ratio):
    """Generates indices to split data into training and test folds. Folds are\
       uniformly distributed within samples. This means that each of the train\
       sets is shifted on the size of the test set from the beginning of the\
       previous train set.

    Args:
        start (int): the first index to start for the first fold
        end (int): the last index to end for the last fold
        num_folds (int): number of folds
        split_ratio (float): split ration between sizes of train and test sets

    Returns:
        generator for indices [train_start, train_end], [test_start, test_end]
    """
    #todo: check parameters:
    # conditions: start>end, num_folds < num_samples, etc.
    #if condition:
    #    raise ValueError("message")

    num_samples = end - start + 1
    test_size = num_samples//(num_folds + split_ratio)
    if test_size == 0:
        test_size = 1
        warnings.warn("Test size is set to 1. Adjust parameters of folds.")

    train_size = num_samples - num_folds*test_size # ajust train size instead
    if train_size < 0:
        raise ValueError("Train size is negative. Adjust parameters of folds.")

    indices = [start, start + train_size, start + train_size + test_size]
    # todo: if numpy is imported use
    # indices = np.array(indices)
    for i in range(num_folds):
        yield [indices[0], indices[1]-1], [indices[1], indices[2]-1]
        indices = list(map(lambda x: x + test_size, indices)) # uniform shift
        # todo: if numpy is imported use
        # indices += test_size


def get_fold_indices(num_samples, num_folds_outer, split_ratio_outer,
                                  num_folds_inner, split_ratio_inner):
    """Generates indices by uniform_loops function for outer and inner loops.\
       For each fold in num_folds_outer yields indices for an outer loop\
       followed by num_folds_inner pairs of indices for inner loop.

    TODO:
        clarify interface for output result (generator, list, etc.)

    Args:
        num_samples (int): number of samples in time series
        num_folds_outer (int): number of outer folds
        split_ratio_outer (float): split ration between sizes of train
                                   and test sets for each outer fold
        num_folds_inner (int): number of inner folds
        split_ratio_inner (float): split ration between sizes of train
                                   and test sets for each inner fold

    Returns:
        generator for indices
    """

    for trn_outer, tst_outer in uniform_loop(0, num_samples-1, num_folds_outer,
                                             split_ratio_outer):
        yield trn_outer, tst_outer
        for trn_inner, tst_inner in uniform_loop(trn_outer[0], trn_outer[-1],
                                           num_folds_inner, split_ratio_inner):
            yield trn_inner, tst_inner


def main():
    num_samples = 100
    num_folds_outer = 5
    split_ratio_outer = 4
    num_folds_inner = 2
    split_ratio_inner = 1

    fold_indices = get_fold_indices(num_samples,
                                    num_folds_outer, split_ratio_outer,
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
    return 0


if __name__ == "__main__":
    main()