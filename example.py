import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

digits = load_digits()

mapper = umap.UMAP().fit(digits.data)
# umap.plot.points(mapper, labels=digits.target)