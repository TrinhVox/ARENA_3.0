# %%
import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"

# %%
arr = np.load(section_dir / "numbers.npy")
print(arr[0].shape)
display_array_as_img(arr[0])
# %%
print(arr[0, 0].shape)
display_array_as_img(arr[0, 0])  # plotting the first channel of the first image, as monochrome

# %%
arr_stacked = einops.rearrange(arr, "b c h w -> c h (b w)")
print(arr_stacked.shape)
display_array_as_img(arr_stacked)  # plotting all images, stacked in a row

# %%
# Exercise 1 - column stacking

arr_column_stacked = einops.rearrange(arr, "b c h w -> c (b h ) w")
print(arr_column_stacked.shape)
display_array_as_img(arr_column_stacked)
# %%
# Exercise 2 - column-stacking and copying

arr_copy_stacked = einops.repeat(arr[0], "c h w -> c (h1 h) w", h1 = 2)
print(arr_copy_stacked.shape)
display_array_as_img(arr_copy_stacked)
# %%
# Exercise 3 - row-stacking and double-copying

arr_row_dcopy = einops.repeat(arr[0:2], " b c h w -> c (b h) (2 w)")
print(arr_row_dcopy.shape)
display_array_as_img(arr_row_dcopy)
# %%

# Exercise 4 - stretching

arr4 = einops.repeat(arr[0], "c h w -> c (h repeat) w", repeat = 2)
display_array_as_img(arr4)
# %%

# Exercise 5 - split channels

arr5 = einops.repeat(arr[0], "c h w -> h (c w)")
display_array_as_img(arr5)
# %%

# Exercise 6 - stack into rows and cols
# Messed up where to place channels - not sure if it should be in front or at the back
arr6 = einops.rearrange(arr, "(b1 b2) c h w  ->  c (b1 h) (b2 w) ", b1=2)

display_array_as_img(arr6)
# %%

# Exercise 7 - transpose

arr7 = einops.rearrange(arr[1], "c h w -> c w h")
display_array_as_img(arr7)
# %%

# Excercise 8 - shrinking 

arr8 = einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", b1 = 2, h2 = 2, w2 = 2)
display_array_as_img(arr8)
# %%
def assert_all_equal(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Tests passed!")


def assert_all_close(actual: Tensor, expected: Tensor, atol=1e-3) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    t.testing.assert_close(actual, expected, atol=atol, rtol=0.0)
    print("Tests passed!")

# %%
# Exercise A1 - rearrange
def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    arr_a1 = einops.rearrange(t.arange(3,9),"(h w) -> h w ", h=3, w= 2)
    return arr_a1

expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)

# %%
# Exercise A2 - rearrange
def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    return einops.rearrange(t.arange(1,7), "(h w) -> h w", h  = 2, w = 3)


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))

# %%
# Exercise B1 - temperature average

def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return einops.reduce(temps, "(h 7) -> h ", "mean")

temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
expected = [71.571, 79.0]
assert_all_close(temperatures_average(temps), t.tensor(expected))

# %%

# Exercise B2 - temperature difference
def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    return temps - einops.repeat(temperatures_average(temps), "h -> (h 7)")


expected = [-0.571, 0.429, -1.571, 3.429, -0.571, 0.429, -1.571, -4.0, 1.0, 6.0, 1.0, -1.0, -7.0, 4.0]
actual = temperatures_differences(temps)
assert_all_close(actual, t.tensor(expected))

# %%
# Exercise B3 - temperature normalized

def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """
    std = einops.reduce(temps, "(h 7) -> h ", t.std)
    return temperatures_differences(temps)/einops.repeat(std, "h -> (h 7)")

expected = [-0.333, 0.249, -0.915, 1.995, -0.333, 0.249, -0.915, -0.894, 0.224, 1.342, 0.224, -0.224, -1.565, 0.894]
actual = temperatures_normalized(temps)
assert_all_close(actual, t.tensor(expected))

# %%
# Exercise C1 - normalize a matrix

def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """

    l2norm = t.norm(matrix, dim=1, keepdim = True)
    return matrix/l2norm




matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[0.267, 0.535, 0.802], [0.456, 0.570, 0.684], [0.503, 0.574, 0.646]])
assert_all_close(normalize_rows(matrix), expected)

# %%

# Exercise C2 - pairwise cosine similarity
def cos_sim_matrix(matrix: Tensor) -> Tensor:
    """Return the cosine similarity matrix for each pair of rows of the given matrix.

    matrix: shape (m, n)
    """
    matrix_normalized = normalize_rows(matrix)
    return matrix_normalized @ matrix_normalized.T


matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[1.0, 0.975, 0.959], [0.975, 1.0, 0.998], [0.959, 0.998, 1.0]])
assert_all_close(cos_sim_matrix(matrix), expected)

# %%


# Exercise D1 - sample distribution

def sample_distribution(probs: Tensor, n: int) -> Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use t.rand and t.cumsum to do this without any explicit loops.
    """
    cumsum = t.cumsum(probs, dim = 0)
    r = t.rand(n, 1)
    mask = r > cumsum
    return mask.sum(dim = -1)
    

n = 5_000_000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs)

# %%

# Exercise E - classifier accuracy

def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use t.argmax.
    """
    correct_scores = t.argmax(scores)
    mask = correct_scores == true_classes
    return sum(mask)/len(scores)


scores = t.tensor([[0.75, 0.5, 0.25], [0.1, 0.5, 0.4], [0.1, 0.7, 0.2]])
true_classes = t.tensor([0, 1, 0])
expected = 2.0 / 3.0
assert classifier_accuracy(scores, true_classes) == expected
print("Tests passed!")

# %%

# Exercise F1 - total price indexing

def total_price_indexing(prices: Tensor, items: Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item.
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    
    """
    return t.gather(prices, 0, items).numpy().sum()


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_indexing(prices, items) == 9.0
print("Tests passed!")


# %%
# Exercise F2 - gather 2D

def gather_2d(matrix: Tensor, indexes: Tensor) -> Tensor:
    """Perform a gather operation along the second dimension.

    matrix: shape (m, n)
    indexes: shape (m, k)

    Return: shape (m, k). out[i][j] = matrix[i][indexes[i][j]]

    For this problem, the test already passes and it's your job to write at least three asserts relating the arguments and the output. This is a tricky function and worth spending some time to wrap your head around its behavior.

    See: https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather
    """
    # YOUR CODE HERE - add assert statement(s) here for `indices` and `matrix`
    assert matrix.ndim == indexes.ndim
    assert indexes.shape[0] <= matrix.shape[0]
    out = matrix.gather(1, indexes)
    # YOUR CODE HERE - add assert statement(s) here for `out`
    assert out.shape == indexes.shape
    return out


matrix = t.arange(15).view(3, 5)
indexes = t.tensor([[4], [3], [2]])
expected = t.tensor([[4], [8], [12]])
assert_all_equal(gather_2d(matrix, indexes), expected)

indexes2 = t.tensor([[2, 4], [1, 3], [0, 2]])
expected2 = t.tensor([[2, 4], [6, 8], [10, 12]])
assert_all_equal(gather_2d(matrix, indexes2), expected2)

# %%
# Exercises F3 - total price gather

def total_price_gather(prices: Tensor, items: Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather."""
    assert items.max() < prices.shape[0]
    return t.gather(prices, 0, items).numpy().sum()

prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_gather(prices, items) == 9.0
print("Tests passed!")

# %%
# Exercises G - indexing
def integer_array_indexing(matrix: Tensor, coords: Tensor) -> Tensor:
    """Return the values at each coordinate using integer array indexing.

    For details on integer array indexing, see:
    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    matrix: shape (d_0, d_1, ..., d_n)
    coords: shape (batch, n)

    Return: (batch, )
    """
    return matrix[tuple(coords.T)]


mat_2d = t.arange(15).view(3, 5)
coords_2d = t.tensor([[0, 1], [0, 4], [1, 4]])
actual = integer_array_indexing(mat_2d, coords_2d)
assert_all_equal(actual, t.tensor([1, 4, 9]))

mat_3d = t.arange(2 * 3 * 4).view((2, 3, 4))
coords_3d = t.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 3], [1, 2, 0]])
actual = integer_array_indexing(mat_3d, coords_3d)
assert_all_equal(actual, t.tensor([0, 5, 10, 15, 20]))

# %%
# Exercise H1 - batch logsumexp - LOOKED AT SOLUTION
def batched_logsumexp(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute log(sum(exp(row))) in a numerically stable way.

    matrix: shape (batch, n)

    Return: (batch, ). For each i, out[i] = log(sum(exp(matrix[i]))).

    Do this without using PyTorch's logsumexp function.

    A couple useful blogs about this function:
    - https://leimao.github.io/blog/LogSumExp/
    - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    raise NotImplementedError()


matrix = t.tensor([[-1000, -1000, -1000, -1000], [1000, 1000, 1000, 1000]])
expected = t.tensor([-1000 + math.log(4), 1000 + math.log(4)])
actual = batched_logsumexp(matrix)
assert_all_close(actual, expected)

matrix2 = t.randn((10, 20))
expected2 = t.logsumexp(matrix2, dim=-1)
actual2 = batched_logsumexp(matrix2)
assert_all_close(actual2, expected2)

# %%
# Exercise H2 - batched softmax

