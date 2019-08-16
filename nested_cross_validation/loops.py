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

    train_size = split_ratio * test_size

    indices = [start, start + train_size, start + train_size + test_size]
    # todo: if numpy is imported use
    # indices = np.array(indices)
    for i in range(num_folds):
        yield [indices[0], indices[1]-1], [indices[1], indices[2]-1]
        indices = list(map(lambda x: x + test_size, indices)) # uniform shift
        # todo: if numpy is imported use
        # indices += test_size