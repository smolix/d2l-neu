#  Preliminaries
:label:`chap_preliminaries`

To prepare for your dive into deep learning,
you will need a few survival skills:
(i) techniques for storing and manipulating data;
(ii) libraries for ingesting 
and preprocessing data from a variety of sources;
(iii) knowledge of the basic linear algebraic operations
that we apply to high-dimensional data elements;
(iv) just enough calculus to determine
which direction to adjust each parameter
in order to decrease the loss function;
(v) the ability to automatically compute derivatives
so that you can forget much of 
the calculus you just learned;
(vi) some basic fluency in probability,
our primary language for reasoning under uncertainty;
and (vii) some aptitude for finding answers 
in the official documentation when you get stuck.

In short, this chapter provides a rapid introduction 
to the basics that you will need to follow 
*most* of the technical content in this book.

```toc
:maxdepth: 2

ndarray
pandas
linear-algebra
calculus
autograd
probability
lookup-api
```

## Resources and Further Reading

The references below round out the survival skills sketched in this chapter---array programming, data wrangling, the linear algebra / calculus / probability we only touch lightly, and the habit of finding answers in official documentation; all are freely accessible online except where noted.

**Books**

- *This book's* **Mathematics for Deep Learning** *part develops the linear algebra, calculus, and probability sketched here in full; start there when you want proper depth on the math the rest of this chapter only previews.*
- [Python for Data Analysis, 3rd ed. — Wes McKinney](https://wesmckinney.com/book/) — free open-access edition by the creator of pandas; the definitive practical guide to NumPy and pandas data manipulation, matching this chapter's ndarray and pandas sections.
- [Python Data Science Handbook — Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/) — free online; a broad, example-driven tour of NumPy, pandas, and the wider scientific-Python stack used throughout the book.
- [Mathematics for Machine Learning — Deisenroth, Faisal & Ong](https://mml-book.github.io/) — free PDF; a complementary, ML-aligned development of the linear algebra, calculus, and probability this chapter only previews.

**Courses and video lectures**

- [NumPy: the absolute basics for beginners — NumPy project](https://numpy.org/doc/stable/user/absolute_beginners.html) — the official getting-started tutorial; the fastest hands-on path to the array operations introduced in the ndarray section.
- [NumPy Tutorials — NumPy project](https://numpy.org/numpy-tutorials/) — official community tutorials as runnable Jupyter notebooks, applying array programming to concrete data tasks.

**Tutorials, notes, and documentation**

- [Array programming with NumPy — Harris, Millman, van der Walt et al. (Nature, 2020)](https://doi.org/10.1038/s41586-020-2649-2) — the reference paper on the n-dimensional array model underpinning every tensor library in this book (open-access; preprint at [arXiv:2006.10256](https://arxiv.org/abs/2006.10256)).
- [NumPy user guide — NumPy project](https://numpy.org/doc/stable/user/index.html) — the official documentation hub; your first stop for looking up array creation, indexing, broadcasting, and dtypes, as urged in the lookup-api section.
- [pandas user guide — pandas project](https://pandas.pydata.org/docs/user_guide/index.html) — the official, topic-organized reference for the loading, indexing, and missing-data handling covered in the pandas section.
- [PyTorch documentation — PyTorch project](https://docs.pytorch.org/docs/stable/index.html) — the official API docs; the canonical place to look up tensor operations and `autograd`, directly reinforcing the lookup-api section's message.
- [SciPy documentation — SciPy project](https://docs.scipy.org/doc/scipy/) — official docs for linear algebra, optimization, and statistics routines that complement the linear-algebra, calculus, and probability sections.

