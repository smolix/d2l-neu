```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Weight Decay
:label:`sec_weight_decay`

In :numref:`sec_generalization_basics` a polynomial of degree 19 fit 20
noisy points with weights of a million or more, and we avoided such fits by
choosing the degree on validation data. Choosing the degree is a choice of
features, and with many inputs it is a rather coarse one: 20 input
variables have 20 features of degree 1, 210 of degree 2, and 1,540 of
degree 3. We would much prefer to limit the flexibility of a model on a
continuous scale, while keeping all of its features.

Where does the excess flexibility of a linear model lie? A small example
shows it. Suppose that the model receives two features that nearly duplicate
each other, such as the same quantity measured by two instruments, and that
the label is close to their sum, $y \approx x_1 + x_2$. The weights $(1, 1)$
fit the training data. The weights $(501, -499)$ fit them almost as well:
they predict $x_1 + x_2 + 500\,(x_1 - x_2)$, and on the training examples
the two measurements differ only slightly. The data say almost nothing about
how to split the weight between the two features. Least squares settles the
split by fitting the noise in the training labels, and since $x_1 - x_2$ is
tiny, moving the predictions even a little through it takes large weights of
opposite sign. The damage shows on new data: on an example whose two
measurements differ by 2, the predictions of the two weight vectors differ
by 1,000. With more features than examples the problem is extreme: 200
weights can fit 20 labels exactly in infinitely many ways, which agree on
the training inputs and disagree on new ones. More data would determine the
weights, but data are often expensive or impossible to collect, and
dropping the second measurement is again a choice of features.

This section keeps all the features and penalizes large weights instead. We
add the squared norm of the weight vector, multiplied by a constant
$\lambda \geq 0$, to the loss. Among weight vectors that fit the data about
equally well, the penalized loss prefers the smallest, $(1, 1)$ in the
example, and $\lambda$ sets the strength of this preference on a continuous
scale. We define the penalty and its effect on gradient descent, try it on
a model with 200 features and only 20 training examples, and then ask which
directions in weight space it shrinks. That question will bring us back to
the duplicated features, where the penalty does its proper work.

```{.python .input #weight-decay}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #weight-decay}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #weight-decay}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #weight-decay}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Norm Penalties

### The Penalized Loss

The loss of linear regression in :numref:`sec_linear_regression` is the
average squared error over the $n$ training examples,

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

To penalize large weights, we add the squared $\ell_2$ norm
$\|\mathbf{w}\|^2 = \sum_j w_j^2$ of :numref:`subsec_lin-algebra-norms`,
scaled by a constant $\lambda \geq 0$, and minimize

$$L(\mathbf{w}, b) + \frac{\lambda}{2}\|\mathbf{w}\|^2.$$
:eqlabel:`eq_wd-objective`

The first term measures the errors on the training data and the second the
size of the weights. The *regularization constant* $\lambda$ weighs one
against the other. With $\lambda = 0$ we recover the original loss. As
$\lambda$ grows, the minimizer moves toward $\mathbf{w} = \mathbf{0}$, the
model that ignores its inputs and predicts $b$ everywhere. Note that the
training loss cannot choose $\lambda$ for us, since it is always lowest at
$\lambda = 0$. Like the degree of a polynomial in
:numref:`subsec_generalization-model-selection`, $\lambda$ is a
hyperparameter, chosen on validation data.

The factor $\frac{1}{2}$ is merely convenient: it makes the gradient of the
penalty $\lambda\mathbf{w}$. Leaving out the bias, by contrast, is
deliberate. Adding a constant to every label should move the fit by that
constant at no cost, whereas a penalty on $b$ would pull every prediction
toward zero rather than toward the average label.

A change to training whose purpose is to lower the error on new data rather
than on the training data is called *regularization*. The penalty
:eqref:`eq_wd-objective` is the most common regularizer in deep learning,
where it is called weight decay
:cite:`Hanson.Pratt.1988,Krogh.Hertz.1992` for a reason we see shortly.
Statisticians call linear regression with this penalty *ridge regression*
:cite:`Hoerl.Kennard.1970`.

What does a smaller norm change about the model? For a linear model, the
norm bounds how strongly the prediction can respond to a change of the
input. By the Cauchy--Schwarz inequality
$|\mathbf{u}^\top\mathbf{v}| \leq \|\mathbf{u}\|\,\|\mathbf{v}\|$ of
:numref:`sec_linear-algebra`, the predictions for two inputs $\mathbf{x}$
and $\mathbf{x}'$ differ by at most

$$|\mathbf{w}^\top \mathbf{x} - \mathbf{w}^\top \mathbf{x}'| \leq \|\mathbf{w}\|\, \|\mathbf{x} - \mathbf{x}'\|.$$

A model with small weights cannot amplify a small change of its input, such
as noise in a measurement. In the opening example, the weights $(1, 1)$
have norm $\sqrt{2} \approx 1.4$ and the weights $(501, -499)$ have norm
about 707, and the second pair turns a difference of 2 between the
measurements into a change of 1,000 in the prediction. Note, however, that
this reading relies on the model being linear, so that its weights are the
slopes of the function it computes. When parameters enter the output in
more complicated ways, as in the networks of later chapters, the same
function can be computed by parameters of very different norms, and the
norm of the parameters is a less direct measure of how much the function
can vary.

### Weight Decay in the Gradient Step

What does the penalty do to training? Adding its gradient
$\lambda\mathbf{w}$ to the minibatch SGD update of
:numref:`subsec_linear-regression-minibatch-sgd`, with learning rate $\eta$
and minibatch $\mathcal{B}_t$, gives

$$\mathbf{w} \leftarrow (1 - \eta\lambda)\,\mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),$$
:eqlabel:`eq_wd-update`

while the update of $b$ does not change. The update first multiplies the
weights by $1 - \eta\lambda$, a number slightly below one, and then takes
the usual step toward fitting the minibatch. Without the data term, the
weights would decay geometrically, to $(1 - \eta\lambda)^t$ times their
initial value after $t$ steps, which is why the method is called *weight
decay*. With $\eta = 0.01$ and $\lambda = 3$, the values of the experiment
below, each step keeps 97% of the weights, and 120 steps keep
$0.97^{120} \approx 2.6\%$. The data term counteracts the decay along
directions in which the training data favor nonzero weights. Along the
other directions nothing stops it.

For minibatch SGD, adding $\frac{\lambda}{2}\|\mathbf{w}\|^2$ to the loss
and multiplying the weights by $1 - \eta\lambda$ in every step are the same
operation, and frameworks choose the second, inside the optimizer (see the
end of this section). The two forms part ways for optimizers that rescale
each coordinate of the gradient, which exercise
:exref:`ex_weight-decay-shrinkage-vs-penalty` explores. An update
that shrinks the weights directly, whatever the optimizer does with the
gradient, is called *decoupled weight decay* :cite:`Loshchilov.Hutter.2019`;
:numref:`sec_adamw` shows why the distinction matters for adaptive
optimizers.

### The $\ell_1$ Penalty and Sparsity

Other norms give other penalties. With the $\ell_1$ norm
$\|\mathbf{w}\|_1 = \sum_j |w_j|$ in place of the squared $\ell_2$ norm,
linear regression becomes *lasso* regression :cite:`Tibshirani.1996`. The
two penalties treat the duplicated features of the opening differently.
When the two measurements agree, the weights $(1, 1)$ and $(2, 0)$ make the
same predictions. The squared $\ell_2$ norm prefers to spread the weight:
it is 2 for $(1, 1)$ and 4 for $(2, 0)$. The $\ell_1$ norm is 2 for both,
and for every split $(a, 2 - a)$ with $0 \leq a \leq 2$, so lasso has no
preference for an even split.

Geometry explains the difference. Minimizing the penalized loss is
equivalent to minimizing $L$ subject to a bound on the norm
($\|\mathbf{w}\| \leq s$ for ridge, $\|\mathbf{w}\|_1 \leq s$ for lasso),
for a bound $s$ that depends on $\lambda$; the appendix derives
this correspondence (:numref:`subsec_mdl-weight-decay-duality`). The
contours of the loss, the sets of weight vectors with equal loss, are
ellipses around the unconstrained minimizer $\hat{\mathbf{w}}$, elongated
along the directions that the data determine poorly, those in which a
change of the weights barely changes the training predictions. The
constrained solution, which we write $\hat{\mathbf{w}}_\lambda$, is the
point where the smallest contour touches the constraint region
(:numref:`fig_ridge_geometry`). The $\ell_2$ ball is round, so the contact
point usually has every coordinate nonzero: ridge regression shrinks the
weights but rarely sets one to zero. The $\ell_1$ ball is a diamond with
its corners on the axes, and a contour often touches it first at a corner,
where some coordinates are zero. Lasso therefore selects features, which
helps when only a few of many features matter, or when measuring each
feature has a cost. The rest of this section uses the $\ell_2$ penalty,
whose minimizer has a closed form.

![Weight decay as a constraint. The contours of the training loss are ellipses around the unconstrained minimizer $\hat{\mathbf{w}}$, and the regularized solution $\hat{\mathbf{w}}_\lambda$ is the point where the smallest contour (orange) touches the constraint region. Left: the $\ell_2$ ball is touched where both coordinates are nonzero. Right: the $\ell_1$ diamond is touched at a corner, where $w_2 = 0$.](../img/mdl-linreg-ridge-geometry.svg)
:label:`fig_ridge_geometry`

### A Gaussian Prior on the Weights

The penalty also has a probabilistic reading, and with it comes an opinion
on how large $\lambda$ should be. In
:numref:`subsec_normal_distribution_and_squared_loss` we derived the squared loss from Gaussian noise. If each label is
$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon$ with
$\epsilon \sim \mathcal{N}(0, \sigma^2)$, the negative log-likelihood of the
training labels is

$$-\log p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}, b) = \frac{1}{2\sigma^2}\sum_{i=1}^n \left(y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)} - b\right)^2 + \textrm{const},$$

and maximum likelihood minimizes it.

Maximum likelihood uses only the data. Suppose that before seeing any data
we believe large weights to be unlikely. We can state this belief as a
*prior* distribution on the weights, here independent normal weights
$w_j \sim \mathcal{N}(0, \tau^2)$, whose standard deviation $\tau$ says
how large we expect them to be.

Bayes' theorem (:numref:`sec_prob`) combines the prior with the likelihood
into the *posterior*
$p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w})\, p(\mathbf{w})$,
the distribution of the weights after seeing the data. The weights with the
highest posterior density are the *maximum a posteriori* (MAP) estimate. In
the negative logarithm of the posterior, the prior contributes

$$-\log p(\mathbf{w}) = \frac{1}{2\tau^2}\|\mathbf{w}\|^2 + \textrm{const},$$

a quadratic penalty on the weights. Multiplying the negative log-posterior
by $\sigma^2/n$, which does not move its minimizer, turns it into the
penalized loss :eqref:`eq_wd-objective` with

$$\lambda = \frac{\sigma^2}{n \tau^2}.$$

Weight decay is thus MAP estimation under a Gaussian prior on the weights,
and the formula says how strong the penalty should be. It grows with the
noise level $\sigma$, since noisy labels deserve less trust, and it grows
as the prior narrows, with smaller $\tau$. It falls as $1/n$, since more
data outweigh the prior. The appendix treats MAP estimation in general
(:numref:`subsec_mdl-gaussian-prior-weight-decay`).

## Weight Decay from Scratch

Let us try the penalty on a model that can overfit. The data come from a
linear model with $d = 200$ standard normal features,

$$y = 0.05 + \sum_{j=1}^{200} 0.01\, x_j + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.01^2),$$

so the true weights are $\mathbf{w}^* = (0.01, \ldots, 0.01)$ and the true
bias is $b^* = 0.05$. We train on only 20 examples, in minibatches of 5,
and validate on 1,000. This is the extreme case of the opening, with ten
times as many weights as training labels.
`SyntheticRegressionData` from :numref:`sec_synthetic-regression-data`
draws such data for any $\mathbf{w}^*$ and $b^*$.

```{.python .input #weight-decay-high-dimensional-linear-regression}
data = d2l.SyntheticRegressionData(w=d2l.ones(200) * 0.01, b=0.05, noise=0.01,
                                   num_train=20, num_val=1000, batch_size=5)
```

### The Penalized Model

The penalty is half the sum of the squared weights.

```{.python .input #weight-decay-defining-ell-2-norm-penalty}
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

We extend the linear regression model of :numref:`sec_linear_scratch`,
which draws its initial weights from $\mathcal{N}(0, 0.01^2)$, a detail
that will matter. Only `training_step` changes: it returns the penalized
loss :eqref:`eq_wd-objective`, which the trainer differentiates and
minimizes. `validation_step` still returns the plain
squared loss, so that validation losses at different values of $\lambda$
all measure the quality of the predictions.

```{.python .input #weight-decay-defining-the-model-1}
%%tab pytorch, tensorflow, jax
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def training_step(self, batch):
        loss = super().training_step(batch)
        return loss + self.lambd * l2_penalty(self.w)
```

```{.python .input #weight-decay-defining-the-model-1}
%%tab mxnet
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def training_step(self, batch):
        loss = super().training_step(batch)
        return loss + self.lambd * l2_penalty(self.w.data())
```

### Training with and without the Penalty

`train_scratch` trains a model with regularization constant `lambd`
(`lambda` is a reserved word in Python) on `data` for 30 epochs at learning
rate 0.01, which makes 120 SGD steps. It returns the model, its final
validation loss, read from `trainer.history`, and the norm of the learned
weights. The plot shows the training and validation losses of each epoch on
a logarithmic scale.

```{.python .input #weight-decay-defining-the-model-2}
%%tab pytorch
def train_scratch(lambd, data):
    trainer = d2l.Trainer(max_epochs=30)
    trainer.board.yscale = 'log'
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    trainer.fit(model, data)
    norm = float(torch.linalg.vector_norm(model.w.detach()))
    return model, trainer.history['val_loss'][-1], norm
```

```{.python .input #weight-decay-defining-the-model-2}
%%tab tensorflow
def train_scratch(lambd, data):
    trainer = d2l.Trainer(max_epochs=30)
    trainer.board.yscale = 'log'
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    trainer.fit(model, data)
    norm = float(tf.norm(model.w))
    return model, trainer.history['val_loss'][-1], norm
```

```{.python .input #weight-decay-defining-the-model-2}
%%tab jax
def train_scratch(lambd, data):
    trainer = d2l.Trainer(max_epochs=30)
    trainer.board.yscale = 'log'
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    trainer.fit(model, data)
    norm = float(jnp.linalg.norm(model.w[...]))
    return model, trainer.history['val_loss'][-1], norm
```

```{.python .input #weight-decay-defining-the-model-2}
%%tab mxnet
def train_scratch(lambd, data):
    trainer = d2l.Trainer(max_epochs=30)
    trainer.board.yscale = 'log'
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    trainer.fit(model, data)
    norm = float(mnp.linalg.norm(model.w.data()))
    return model, trainer.history['val_loss'][-1], norm
```

We start without the penalty.

```{.python .input #weight-decay-training-without-regularization}
model_0, val_loss, norm = train_scratch(0, data)
```

```{.python .input #weight-decay-training-with-and-without-the-penalty-1}
print(f'validation loss {val_loss:.4f}, norm of w {norm:.3f}')
```

The training loss falls by many orders of magnitude, to below $10^{-7}$:
the model fits all 20 training labels. The validation loss barely moves
and stays near 0.02, between about 0.015 and 0.025 from run to run. A
perfect fit of 20 labels has taught the model very little.

Now we add the penalty. Which $\lambda$? The prior has an opinion: with
$\sigma = 0.01$, $n = 20$, and weights of size $\tau = 0.01$, it asks for
$\lambda = \sigma^2/(n\tau^2) = 0.05$. We use $\lambda = 3$, sixty times as
much, and owe the reader the reason. The formula describes the minimizer of
the penalized loss, not the point that 120 steps of SGD reach. At
$\lambda = 0.05$ the whole run would keep $0.9995^{120} \approx 94\%$ of
the weights, and the penalty would leave almost no trace (exercise
:exref:`ex_weight-decay-map` asks which prior our value implies).

```{.python .input #weight-decay-using-weight-decay}
model_3, val_loss, norm = train_scratch(3, data)
```

```{.python .input #weight-decay-training-with-and-without-the-penalty-2}
print(f'validation loss {val_loss:.4f}, norm of w {norm:.3f}')
```

The training curve now shows the penalized loss, which levels off at about
$10^{-3}$ to $10^{-2}$, since small weights cannot fit the training labels
exactly. The validation loss falls steadily, to about 0.01, roughly half
its value without the penalty, and the norm of the weights shrinks from
between 0.12 and 0.16 to between 0.02 and 0.06. Each run draws new data,
so the numbers vary from run to run, but these comparisons do not. Has the
penalty cured the overfitting? It is too early to say so.

## Shrinkage Along Eigenvectors
:label:`subsec_wd-shrinkage`

What did the penalty remove? For linear regression we can answer exactly:
the minimizer of the penalized loss has a closed form, which shows the
parts of the weights that the penalty shrinks. It does two things. Along
directions in weight space that the data determine poorly, it damps the
large coordinates that least squares picks up from noise in the labels.
Along directions that the data do not determine at all, it removes whatever
the weights hold there, including what the initialization put in. The
first effect is the one that makes weight decay worth having, and a small
example below shows it. The second, as it turns out, is all that helped in
our experiment.

### The Ridge Estimator

To keep the formulas short, we drop the bias in this section. With an
unpenalized bias, the same analysis applies after the features and the
labels are centered (exercise :exref:`ex_linear-regression-centering`).
Without the bias, the
loss is $L(\mathbf{w}) = \frac{1}{2n}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$,
with gradient $\frac{1}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$.
Adding the gradient $\lambda\mathbf{w}$ of the penalty and setting the sum
to zero gives

$$\left(\frac{1}{n}\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}\right)\mathbf{w} = \frac{1}{n}\mathbf{X}^\top\mathbf{y}.$$
:eqlabel:`eq_wd-ridge-normal`

At $\lambda = 0$ these are the normal equations of
:numref:`subsec_linear-regression-analytic-solution`, divided by $n$. They
have many solutions when the columns of $\mathbf{X}$ are dependent, as they
are for 20 examples with 200 features. For $\lambda > 0$ the matrix on the
left is invertible: for every $\mathbf{v} \neq \mathbf{0}$,
$\mathbf{v}^\top\left(\frac{1}{n}\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}\right)\mathbf{v} = \frac{1}{n}\|\mathbf{X}\mathbf{v}\|^2 + \lambda\|\mathbf{v}\|^2 > 0$,
so the matrix maps no nonzero vector to zero. The penalized loss therefore
has exactly one minimizer, the *ridge estimator* $\hat{\mathbf{w}}_\lambda$.

### Shrinkage Factors

The matrix $\frac{1}{n}\mathbf{X}^\top\mathbf{X}$ is symmetric, so it has
perpendicular eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_d$ of length 1,
one per dimension, with real eigenvalues $\mu_1, \ldots, \mu_d$
(:numref:`subsec_lin-alg-eigenvalues`). Each eigenvalue measures how
strongly the training data determine the weights along its eigenvector.
Since
$\mu_j = \mathbf{v}_j^\top \left(\frac{1}{n}\mathbf{X}^\top\mathbf{X}\right) \mathbf{v}_j = \frac{1}{n}\|\mathbf{X}\mathbf{v}_j\|^2$,
moving the weights a unit distance along $\mathbf{v}_j$ changes the
predictions on the training inputs by $\mu_j$ in mean square. The
eigenvalues are never negative, and $\mu_j = 0$ means that the move changes
no training prediction at all.

In the basis of eigenvectors, the equations :eqref:`eq_wd-ridge-normal`
separate. We write the weights as $\mathbf{w} = \sum_j c_j \mathbf{v}_j$
and the right-hand side as
$\frac{1}{n}\mathbf{X}^\top\mathbf{y} = \sum_j g_j \mathbf{v}_j$. The
matrix maps $\mathbf{v}_j$ to $(\mu_j + \lambda)\,\mathbf{v}_j$, so the
equations become $(\mu_j + \lambda)\,c_j = g_j$, one for each $j$. Without
the penalty, least squares gives $c_j = g_j/\mu_j$ wherever $\mu_j > 0$.
With the penalty,

$$c_j = \frac{\mu_j}{\mu_j + \lambda} \cdot \frac{g_j}{\mu_j}.$$

The ridge estimator multiplies the least-squares coordinate along each
eigenvector by the *shrinkage factor* $\mu_j/(\mu_j + \lambda)$.

The factor is close to 1 when $\mu_j \gg \lambda$, close to 0 when
$\mu_j \ll \lambda$, and $\frac{1}{2}$ at $\mu_j = \lambda$
(:numref:`fig_ridge_shrinkage`). Directions that the data determine well
keep their coordinates, and directions that they determine poorly are
suppressed. In the poorly determined directions least squares is easily
misled: noise in the labels changes $g_j$ a little, and the least-squares
coordinate $g_j/\mu_j$ a lot when $\mu_j$ is small.

![The shrinkage factor $\mu/(\mu + \lambda)$ of the ridge estimator along an eigenvector with eigenvalue $\mu$, against the regularization constant $\lambda$. A direction keeps its coordinate while $\lambda \ll \mu$ and loses it once $\lambda \gg \mu$; the factor is $\frac{1}{2}$ at $\lambda = \mu$ (dots). Directions with $\mu = 0$ are set to zero for every $\lambda > 0$ (dashed).](../img/mdl-linreg-ridge-shrinkage.svg)
:label:`fig_ridge_shrinkage`

Directions with $\mu_j = 0$ are the extreme case. There
$g_j = \frac{1}{n}\mathbf{v}_j^\top\mathbf{X}^\top\mathbf{y} = \frac{1}{n}(\mathbf{X}\mathbf{v}_j)^\top\mathbf{y} = 0$,
so the least-squares equation $\mu_j c_j = g_j$ reads $0 = 0$ and leaves
$c_j$ undetermined, while the ridge estimator sets $c_j = 0$ for every
$\lambda > 0$.

Adding up the shrinkage factors gives the *effective number of parameters*
(statisticians say degrees of freedom),

$$\textrm{df}(\lambda) = \sum_{j=1}^d \frac{\mu_j}{\mu_j + \lambda}.$$

A direction counts as one parameter if it is kept, as none if it is
suppressed, and as a fraction in between. As $\lambda$ decreases to 0,
$\textrm{df}$ approaches the number of nonzero eigenvalues, and as
$\lambda$ grows it falls continuously to 0. This is the continuous control
of capacity we asked for at the start of the section. Instead of adding or
removing whole features, $\lambda$ moves the effective number of parameters
through every value in between.

### Two Duplicated Features

We can now return to the duplicated features of the opening and watch this
damping with numbers. We draw $n = 100$ examples with $x_1 \sim \mathcal{N}(0, 1)$ and a second
feature $x_2 = x_1 + 0.001\,z$ with $z \sim \mathcal{N}(0, 1)$,
label them $y = x_1 + x_2 + \epsilon$ with $\epsilon \sim \mathcal{N}(0, 0.1^2)$,
and fit without a bias. The sum direction $(1, 1)/\sqrt{2}$ then has an
eigenvalue between about 1 and 3, and the difference direction
$(1, -1)/\sqrt{2}$ one below $10^{-6}$, so at $\lambda = 0.01$ ridge keeps
the sum and removes the difference. NumPy suffices here: it draws the data
and solves both problems in closed form. We test each fit on 1,000 new
examples drawn in the same way and on 1,000 whose features disagree more,
$x_2 = x_1 + 0.1\,z$.

```{.python .input #weight-decay-two-duplicated-features-1}
rng = np.random.default_rng()  # A new draw on every run

def twins(n, gap):
    """Two near-copies of a feature, x2 = x1 + gap * noise, labeled by
    y = x1 + x2 + noise with standard deviation 0.1."""
    x1 = rng.normal(size=n)
    X = np.stack([x1, x1 + gap * rng.normal(size=n)], axis=1)
    return X, X.sum(axis=1) + 0.1 * rng.normal(size=n)

def mse(w, data):
    """Mean squared error of the weights w on data = (X, y)."""
    X, y = data
    return ((X @ w - y) ** 2).mean()
```

We fit two training sets, each by least squares and by ridge regression.

```{.python .input #weight-decay-two-duplicated-features-2}
lambd = 0.01
near, far = twins(1000, 0.001), twins(1000, 0.1)  # the two test sets
for _ in range(2):                                # two training sets
    X, y = twins(100, 0.001)
    w_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    w_ridge = np.linalg.solve(X.T @ X / len(X) + lambd * np.eye(2),
                              X.T @ y / len(X))
    for name, w in (('least squares', w_ls), ('ridge', w_ridge)):
        print(f'{name:>13} ({w[0]:6.2f}, {w[1]:6.2f}): '
              f'test {mse(w, near):.3f}, disagreeing {mse(w, far):.3f}')
mu = np.linalg.eigvalsh(X.T @ X / len(X))         # in increasing order
print(f'eigenvalues of X^T X / n: {mu[0]:.1e}, {mu[1]:.1e}')
```

Least squares returns different weights for each training set. They always
add up to about 2, the sum that the data determine, but the split between
them changes from draw to draw, with weights often several units away from
1 and of opposite sign. Ridge returns about $(1, 1)$ every time. On new
examples like the training examples the difference is invisible: both fits
have a test error of about 0.01, the noise variance $\sigma^2$. It shows on
examples whose features disagree more, where least squares fails whenever
its weights lie far from 1. Since $w_2 - 1 \approx -(w_1 - 1)$, its error
on such an example is about $(w_1 - 1)(x_1 - x_2)$ plus the noise, with
mean square about $0.01\,(w_1 - 1)^2 + 0.01$. The large weights amplify the
disagreement, just as the Cauchy--Schwarz bound allows. Ridge keeps its
error at about 0.01. This is the situation the penalty is made for: a
direction that the data barely determine, and noise ready to fill it.

### The Directions Our Data Determine

Was the experiment with 200 features such a situation? To find out, we
compute the eigenvalues of $\frac{1}{n}\mathbf{X}^\top\mathbf{X}$ for its
20 training inputs (uncentered, which changes the numbers only slightly),
and the shrinkage factors at $\lambda = 3$. As in
:numref:`sec_linear-algebra`, `eigh` returns the eigenvalues in increasing
order and the eigenvectors as the columns of its second result.

```{.python .input #weight-decay-why-shrinkage-helps-the-spectral-view}
%%tab pytorch
X = data.X[:data.num_train]                   # the 20 training inputs
mu, V = torch.linalg.eigh(X.T @ X / len(X))   # increasing eigenvalues
mu_data = mu[-len(X):]                        # the 20 largest
factors = mu_data / (mu_data + 3)             # shrinkage at lambda = 3
print(f'the other 180 eigenvalues are at most {float(mu[-len(X) - 1]):.0e}')
print(f'mu from {float(mu_data[0]):.1f} to {float(mu_data[-1]):.1f}, '
      f'factors from {float(factors[0]):.2f} to {float(factors[-1]):.2f}, '
      f'df = {float(factors.sum()):.1f}')
```

```{.python .input #weight-decay-why-shrinkage-helps-the-spectral-view}
%%tab tensorflow
X = data.X[:data.num_train]                   # the 20 training inputs
mu, V = tf.linalg.eigh(tf.transpose(X) @ X / len(X))  # increasing
mu_data = mu[-len(X):]                        # the 20 largest
factors = mu_data / (mu_data + 3)             # shrinkage at lambda = 3
print(f'the other 180 eigenvalues are at most {float(mu[-len(X) - 1]):.0e}')
print(f'mu from {float(mu_data[0]):.1f} to {float(mu_data[-1]):.1f}, '
      f'factors from {float(factors[0]):.2f} to {float(factors[-1]):.2f}, '
      f'df = {float(tf.reduce_sum(factors)):.1f}')
```

```{.python .input #weight-decay-why-shrinkage-helps-the-spectral-view}
%%tab jax
X = data.X[:data.num_train]                   # the 20 training inputs
mu, V = jnp.linalg.eigh(X.T @ X / len(X))     # increasing eigenvalues
mu_data = mu[-len(X):]                        # the 20 largest
factors = mu_data / (mu_data + 3)             # shrinkage at lambda = 3
print(f'the other 180 eigenvalues are at most {float(mu[-len(X) - 1]):.0e}')
print(f'mu from {float(mu_data[0]):.1f} to {float(mu_data[-1]):.1f}, '
      f'factors from {float(factors[0]):.2f} to {float(factors[-1]):.2f}, '
      f'df = {float(factors.sum()):.1f}')
```

```{.python .input #weight-decay-why-shrinkage-helps-the-spectral-view}
%%tab mxnet
X = data.X[:data.num_train]                   # the 20 training inputs
mu, V = mnp.linalg.eigh(X.T @ X / len(X))     # increasing eigenvalues
mu_data = mu[-len(X):]                        # the 20 largest
factors = mu_data / (mu_data + 3)             # shrinkage at lambda = 3
print(f'the other 180 eigenvalues are at most {float(mu[-len(X) - 1]):.0e}')
print(f'mu from {float(mu_data[0]):.1f} to {float(mu_data[-1]):.1f}, '
      f'factors from {float(factors[0]):.2f} to {float(factors[-1]):.2f}, '
      f'df = {float(factors.sum()):.1f}')
```

Only the 20 largest eigenvalues are nonzero, between about 4 and 19. The
other 180 are zero up to rounding errors of a few times $10^{-6}$: 20
examples cannot determine more than 20 directions in a 200-dimensional
weight space. At $\lambda = 3$ the shrinkage factors of the 20 determined
directions lie between about 0.6 and 0.85, and $\textrm{df}(3) \approx 15$.
The regularized model has 200 weights but about 15 effective parameters.

The 180 undetermined directions explain the experiment. Gradient descent
never changes the weights along them. The gradient of the unpenalized loss,
$\frac{1}{n}\mathbf{X}^\top\mathbf{r}$ with the errors
$\mathbf{r} = \mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}$, is a
combination of the training inputs, and every training input is
perpendicular to every $\mathbf{v}_j$ with
$\mathbf{X}\mathbf{v}_j = \mathbf{0}$. Without the penalty, whatever the
random initialization put into these directions stays there. With it, every
step multiplies that part by 0.97. We split the trained weights of both
models into their part in the 20 determined directions and the rest. Since
the columns of `V_data` are perpendicular and of length 1, `V_data.T @ w`
holds the coordinates of $\mathbf{w}$ along them, and
`V_data @ (V_data.T @ w)` is the projection of $\mathbf{w}$ onto their span.

```{.python .input #weight-decay-the-directions-our-data-determine-1}
%%tab pytorch
V_data = V[:, -len(X):]                       # directions the data determine
for lambd, model in ((0, model_0), (3, model_3)):
    w = model.w.detach()
    w_data = V_data @ (V_data.T @ w)
    print(f'lambda = {lambd}: '
          f'determined part {float(torch.linalg.vector_norm(w_data)):.3f}, '
          f'rest {float(torch.linalg.vector_norm(w - w_data)):.3f}')
```

```{.python .input #weight-decay-the-directions-our-data-determine-1}
%%tab tensorflow
V_data = V[:, -len(X):]                       # directions the data determine
for lambd, model in ((0, model_0), (3, model_3)):
    w = model.w
    w_data = V_data @ (tf.transpose(V_data) @ w)
    print(f'lambda = {lambd}: '
          f'determined part {float(tf.norm(w_data)):.3f}, '
          f'rest {float(tf.norm(w - w_data)):.3f}')
```

```{.python .input #weight-decay-the-directions-our-data-determine-1}
%%tab jax
V_data = V[:, -len(X):]                       # directions the data determine
for lambd, model in ((0, model_0), (3, model_3)):
    w = model.w[...]
    w_data = V_data @ (V_data.T @ w)
    print(f'lambda = {lambd}: '
          f'determined part {float(jnp.linalg.norm(w_data)):.3f}, '
          f'rest {float(jnp.linalg.norm(w - w_data)):.3f}')
```

```{.python .input #weight-decay-the-directions-our-data-determine-1}
%%tab mxnet
V_data = V[:, -len(X):]                       # directions the data determine
for lambd, model in ((0, model_0), (3, model_3)):
    w = model.w.data()
    w_data = V_data @ (V_data.T @ w)
    print(f'lambda = {lambd}: '
          f'determined part {float(mnp.linalg.norm(w_data)):.3f}, '
          f'rest {float(mnp.linalg.norm(w - w_data)):.3f}')
```

Without the penalty, the undetermined part has norm about 0.13 (0.11 to
0.15 from run to run). This is simply the random initialization in 180
directions, $0.01\sqrt{180} \approx 0.134$, which training never touched.
For a new input $\mathbf{x}$ with standard normal features, this part adds
its dot product with $\mathbf{x}$ to the prediction, an error with variance
equal to its squared norm, about 0.018. With the factor $\frac{1}{2}$ of our
loss, that is 0.009, roughly the difference between the two validation
losses. With $\lambda = 3$, 120 steps of decay leave 2.6% of the
undetermined part, a norm between 0.003 and 0.004. The determined part
shrinks too, by the factors computed above, but this does not help here.
All 20 of its eigenvalues are 4 or larger, so the label noise, with
$\sigma = 0.01$, moves the least-squares coordinates very little. Nothing
in this problem resembles the difference direction of the duplicated
features.

We should be candid about how little was gained. The penalty cannot recover
what the data do not determine. About 180/200 of
$\|\mathbf{w}^*\|^2 = 0.02$, or 0.018, lies in the undetermined directions,
so a model that sets its undetermined part to zero, as weight decay does,
has a validation loss of at least about $\frac{1}{2} \cdot 0.018 = 0.009$.
The regularized model, at about 0.01, comes close to this limit. So,
however, does a model that ignores its inputs and predicts $b^* = 0.05$
everywhere, whose loss is $\frac{1}{2}(0.02 + \sigma^2) \approx 0.01$. And
so does the unpenalized model started from $\mathbf{w} = \mathbf{0}$: its
undetermined part stays zero (exercise
:exref:`ex_weight-decay-starting-point`), and gradient descent on the
squared loss started from zero converges, when it converges at all, to the
interpolating solution of smallest norm
(:numref:`subsec_implicit_regularization`). In this experiment, then,
weight decay did one thing: it removed the random initialization from the
directions that 20 examples cannot see. Its other effect, damping the
directions that the data determine poorly, is what the duplicated features
showed, and no choice of starting point provides it.

## Concise Implementation

Weight decay is so common that deep learning frameworks build it into
their optimizers. The optimizer adds $\lambda\mathbf{w}$ to the gradient of
each weight, or equivalently multiplies the weight by $1 - \eta\lambda$, in
the same pass in which it updates the weight, so no penalty needs to
appear in the loss. Frameworks also let different parameters follow
different rules, which is how we decay the weights but not the bias.

:begin_tab:`mxnet`
Gluon's `Trainer` takes a weight decay `wd`, which each parameter scales by
its own `wd_mult`. Setting `wd_mult = 0` for the bias exempts it.
:end_tab:

:begin_tab:`pytorch`
PyTorch optimizers take a `weight_decay` argument. Passing the parameters
in two groups decays the weight `net.weight` but not the bias `net.bias`.
As in :numref:`sec_linear_concise`, we initialize the weights from
$\mathcal{N}(0, 0.01^2)$ and the bias with zero.
:end_tab:

:begin_tab:`tensorflow`
Keras optimizers take a `weight_decay` argument and apply it in the second
form, multiplying each variable by $1 - \eta \cdot$ `weight_decay` before
every update. `exclude_from_weight_decay` exempts the bias.
:end_tab:

:begin_tab:`jax`
`optax.sgd` has no weight decay, but optax builds optimizers by chaining
transformations of the gradient. `add_decayed_weights` adds `wd` times the
parameter to its gradient, and `masked` applies it only to the parameters
that a mask selects: here those with more than one dimension, the kernel
but not the bias.
:end_tab:

```{.python .input #weight-decay-concise-implementation-1}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()

    def configure_optimizers(self):
        for p in self.collect_params('.*bias').values():
            p.wd_mult = 0
        return gluon.Trainer(self.collect_params(), 'sgd',
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input #weight-decay-concise-implementation-1}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```

```{.python .input #weight-decay-concise-implementation-1}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = tf.keras.optimizers.SGD(self.lr, weight_decay=self.wd)
        optimizer.exclude_from_weight_decay(var_names=['bias'])
        return optimizer
```

```{.python .input #weight-decay-concise-implementation-1}
%%tab jax
def is_matrix(params):
    """Mask that selects the parameters with more than one dimension."""
    return jax.tree.map(lambda p: p.ndim > 1, params)

class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, num_inputs, lr, rngs=None):
        super().__init__(num_inputs, lr, rngs)
        self.save_hyperparameters(ignore=['rngs'])

    def configure_optimizers(self):
        return optax.chain(
            optax.masked(optax.add_decayed_weights(self.wd), is_matrix),
            optax.sgd(self.lr))
```

To reproduce `train_scratch(3, data)` we have to convert both
hyperparameters. The framework loss lacks our factor $\frac{1}{2}$, so we
halve the learning rate to 0.005, as in :numref:`sec_linear_concise`. The
decay term $\lambda\mathbf{w}$, however, comes from the optimizer and is
not doubled along with the gradient of the loss. At half the learning rate
the weight decay must therefore double, to 6, for each step to keep the
factor $1 - 0.005 \cdot 6 = 0.97$.

```{.python .input #weight-decay-concise-implementation-2}
%%tab pytorch
trainer = d2l.Trainer(max_epochs=30)
trainer.board.yscale = 'log'
model = WeightDecay(wd=6, lr=0.005)
model(data.X[:1])  # Creates the parameters of the lazy layer
nn.init.normal_(model.net.weight, std=0.01)
nn.init.zeros_(model.net.bias)
trainer.fit(model, data)
norm = float(torch.linalg.vector_norm(model.get_w_b()[0]))
```

```{.python .input #weight-decay-concise-implementation-2}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=30)
trainer.board.yscale = 'log'
model = WeightDecay(wd=6, lr=0.005)
trainer.fit(model, data)
norm = float(tf.norm(model.get_w_b()[0]))
```

```{.python .input #weight-decay-concise-implementation-2}
%%tab jax
trainer = d2l.Trainer(max_epochs=30)
trainer.board.yscale = 'log'
model = WeightDecay(wd=6, num_inputs=200, lr=0.005)
trainer.fit(model, data)
norm = float(jnp.linalg.norm(model.get_w_b()[0]))
```

```{.python .input #weight-decay-concise-implementation-2}
%%tab mxnet
trainer = d2l.Trainer(max_epochs=30)
trainer.board.yscale = 'log'
model = WeightDecay(wd=6, lr=0.005)
trainer.fit(model, data)
norm = float(mnp.linalg.norm(model.get_w_b()[0]))
```

```{.python .input #weight-decay-concise-implementation-3}
print(f'validation loss {trainer.history["val_loss"][-1]:.4f}, '
      f'norm of w {norm:.3f}')
```

The norm of the weights matches the scratch run at $\lambda = 3$, and so
does the validation loss: near 0.02, it is the framework's mean squared
error, twice our loss. The training curve now shows the plain mean squared
error rather than the penalized loss, because the decay lives in the
optimizer, not in the loss.

In practice, weight decay is this one optimizer argument, with $\lambda$
chosen on validation data, one training run per candidate. What the
argument controls in a deep network, whose parameters are no longer the
slopes of the function, is a harder question, and the bound for linear
models does not answer it.

## Summary

Weight decay adds the penalty $\frac{\lambda}{2}\|\mathbf{w}\|^2$ to the
loss, so that among weight vectors that fit the data about equally well,
training prefers the smallest. Under minibatch SGD the penalty multiplies
the weights by $1 - \eta\lambda$ in every step, and frameworks implement it
inside the optimizer rather than in the loss. The $\ell_1$ penalty of
lasso sets weights to zero where the $\ell_2$ penalty of ridge regression
shrinks them, and the $\ell_2$ penalty is MAP estimation under a Gaussian prior,
with $\lambda = \sigma^2/(n\tau^2)$. For linear regression, the penalty
multiplies the coordinate along each eigenvector of
$\frac{1}{n}\mathbf{X}^\top\mathbf{X}$ by $\mu_j/(\mu_j + \lambda)$.
Directions that the data determine well survive, the others are damped or
removed, and $\lambda$ adjusts the effective number of parameters
continuously. With 200 features and 20 training examples, weight decay
roughly halved the validation loss, but only by removing the random initialization from the 180 directions that the
examples cannot see, which a start from zero achieves as well. The damping
is what no starting point provides: for two duplicated features, ridge
returned weights near $(1, 1)$ where least squares returned large weights
of opposite sign.

## Exercises

1. **The $\ell_1$ update.** Replace the penalty
   $\frac{\lambda}{2}\|\mathbf{w}\|^2$ by $\lambda \sum_j |w_j|$.
    1. Derive the minibatch SGD update for $\mathbf{w}$, using the
       subgradient $\operatorname{sign}(w_j)$ of
       :numref:`sec_linear_regression` in place of the derivative of
       $|w_j|$.
    1. Compare the change that the penalty alone makes to $w_j$ in one step
       of this update with the change it makes in the update
       :eqref:`eq_wd-update`, once for large and once for small $|w_j|$.
    1. The update of (a) almost never sets $w_j$ exactly to zero. Split it
       into a gradient step on $L$ followed by the penalty step, and modify
       the penalty step so that it sets $w_j$ to zero whenever it would
       change the sign of $w_j$. Relate the result to the feature selection
       that lasso performs. The modified step is *soft-thresholding*, the
       proximal step of the $\ell_1$ norm
       (:numref:`subsec_mdl-proximal-operators`).

1. **Shrinkage versus penalty.** Weight decay can be applied by shrinking
   the weights before the gradient step or by adding
   $\frac{\lambda}{2}\|\mathbf{w}\|^2$ to the loss $L$.
    1. Show that the update
       $\mathbf{w} \leftarrow (1-\eta\lambda)\mathbf{w} - \eta\nabla L$ is
       identical to a gradient step on the penalized loss
       $L + \frac{\lambda}{2}\|\mathbf{w}\|^2$.
    1. Suppose that the gradient is rescaled coordinate-wise by fixed
       constants $a_j > 0$ before the step. Shrinking and then stepping
       gives the first update below, and stepping on the penalized gradient
       gives the second:

       $$w_j \leftarrow (1-\eta\lambda)\, w_j - \eta a_j \,\partial_{w_j} L
       \qquad \textrm{and} \qquad
       w_j \leftarrow w_j - \eta a_j \left(\partial_{w_j} L + \lambda w_j\right).$$

       Determine for which $a_j$ the two coincide, and the factor by which
       the second update shrinks $w_j$ in a step with
       $\partial_{w_j} L = 0$. Many optimizers used in deep learning
       rescale the gradient in this way. Which of the two updates keeps
       $\lambda$ a rate of decay that is the same for every weight?

    *Adapted from Simon Prince,
    [Understanding Deep Learning](https://udlbook.github.io/udlbook/),
    Problem 9.5.*
   :exlabel:`ex_weight-decay-shrinkage-vs-penalty`

1. **MAP estimation.** For the prior
   $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$ and noise
   variance $\sigma^2$, this section derived
   $\lambda = \sigma^2/(n\tau^2)$.
    1. For `train_scratch(3, data)` ($n = 20$, $\sigma = 0.01$,
       $\lambda = 3$), compute $\tau$ and compare it with the true weights $w^*_j = 0.01$.
       Is the implied prior consistent with the process that generated the
       data? Compute the $\lambda$ that a prior with $\tau = 0.01$ implies.
    1. Holding $\tau$ and $\sigma$ fixed, describe how $\lambda$ changes as
       $n$ grows, and interpret the change in terms of the relative weight
       of prior and likelihood.
    1. The bias $b$ is not penalized. Which prior on $b$ does this
       correspond to?
   :exlabel:`ex_weight-decay-map`

1. [code] **Ridge as augmented least squares.** Center the 20 training
   inputs and labels so that the bias drops out, and compute the ridge
   estimator for $\lambda = 3$ in two ways.
    1. Solve the linear system :eqref:`eq_wd-ridge-normal`.
    1. Run ordinary least squares with `lstsq` on $\mathbf{X}$ stacked on
       top of $\sqrt{n\lambda}\,\mathbf{I}$, with the labels $\mathbf{y}$
       followed by $d$ zeros.

    Confirm that the two agree up to rounding errors, and explain why. Then
    compare the norm of the result with the norm that
    `train_scratch(3, data)` returned, and name the differences between the two procedures that
    could account for any gap.

    *Adapted from Hastie, Tibshirani, and Friedman,
    [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/),
    Exercise 3.12.*
   :exlabel:`ex_weight-decay-augmented-least-squares`

1. [code] **The $\lambda$ sweep.** Train `WeightDecayScratch` on `data`
   for $\lambda \in \{0.01, 0.1, 0.3, 1, 3, 10, 30\}$.
    1. Plot the final validation loss and the norm of $\mathbf{w}$ against
       $\lambda$ on a logarithmic axis, with the values of
       `train_scratch(0, data)` as horizontal lines, and report the $\lambda$ with the smallest
       validation loss.
    1. Compute $\textrm{df}(\lambda)$ for the same values from the
       eigenvalues of :numref:`subsec_wd-shrinkage`, and plot the
       validation loss against $\textrm{df}(\lambda)$. Compare the shape
       with :numref:`fig_capacity_vs_error`.
    1. Exercise :exref:`ex_weight-decay-map` suggests $\lambda = 0.05$.
       Compare the validation loss of the ridge estimator at
       $\lambda = 0.05$, computed as in exercise
       :exref:`ex_weight-decay-augmented-least-squares`, with that of
       `WeightDecayScratch` trained for 30 epochs at the same $\lambda$.
       Explain the difference with the decay factor
       $(1 - \eta\lambda)^t$, and estimate how many epochs training would
       need to close the gap.

1. [code] **Stopping early versus decay.** Train `WeightDecayScratch` with
   $\lambda = 0$ for 30 epochs and find the epoch with the lowest
   validation loss in `trainer.history`. Compare that loss with the final
   validation losses of `train_scratch(0, data)` and
   `train_scratch(3, data)`. Use the split of the weights in
   :numref:`subsec_wd-shrinkage` to explain why stopping early cannot do
   here what weight decay does.

1. [code] **Duplicated features.** Use the functions `twins` and `mse`
   of :numref:`subsec_wd-shrinkage`.
    1. Sweep $\lambda$ over $10^{-8}, 10^{-7}, \ldots, 10$ and plot both
       ridge weights and both test errors against $\lambda$ on a
       logarithmic axis. For which range of $\lambda$ does ridge do well on
       both test sets? Relate the two ends of the range to the two
       eigenvalues of $\frac{1}{n}\mathbf{X}^\top\mathbf{X}$.
    1. Run gradient descent on the unpenalized loss from $\mathbf{w} = \mathbf{0}$
       with learning rate 0.1 for $10^2$, $10^4$, and $10^6$ steps. Compare
       the weights and test errors with those of least squares and ridge,
       and explain the result with the factor $(1 - \eta\mu_j)^t$ of each
       eigenvector.
    1. Fit least squares to labels without noise, $y = x_1 + x_2$. What
       changes, and why?
   :exlabel:`ex_weight-decay-duplicated-features`

1. [code] **Starting point.** ● `WeightDecayScratch` draws its initial
   weights from $\mathcal{N}(0, 0.01^2)$.
    1. Train with $\lambda = 0$ and with $\lambda = 3$ starting from
       $\mathbf{w} = \mathbf{0}$ (pass `sigma=0`). Before you run, predict
       both validation losses and the norms of both parts of the weights
       from the split in :numref:`subsec_wd-shrinkage`; then compare.
    1. Is the part of $\mathbf{w}$ in the undetermined directions exactly
       unchanged by training at $\lambda = 0$, or only approximately?
       Record the initial weights of a model, train it, and measure.
       Consider minibatch SGD, the bias, and rounding.
    1. Which part of the benefit of weight decay in this section's
       experiment comes from the initialization, and which part would
       remain for data with strongly correlated features, as in exercise
       :exref:`ex_weight-decay-duplicated-features`?
   :exlabel:`ex_weight-decay-starting-point`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/236)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17979)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.7]{.kicker}

A penalty on large weights<br>
**the ℓ2 penalty · the decay step · lasso and the Gaussian prior · shrinkage along eigenvectors**
:::
:::

::: {.slide title="Near-Duplicate Features Admit Weights (1, 1) and (501, −499)"}
[The problem]{.kicker}

Two features measure the same quantity; the label is
$y \approx x_1 + x_2$.

- $(1, 1)$ and $(501, -499)$ both fit the training data: the second adds
  $500\,(x_1 - x_2)$, and $x_1 \approx x_2$ there.
- On a new example with $x_1 - x_2 = 2$, their predictions differ by
  **1,000**.
- 200 weights, 20 examples: infinitely many exact fits.
- Dropping features is coarse: 20 variables give 20, 210, and 1,540
  monomials of degree 1, 2, and 3.

::: {.d2l-note}
Keep every feature; penalize large weights, with a strength $\lambda$ we
can set continuously.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write the penalized loss and its SGD update, and say why it is called
  weight decay;
- tell ridge from lasso by the shape of the constraint region;
- derive $\lambda$ from a Gaussian prior on the weights;
- compute the shrinkage factor $\mu/(\mu + \lambda)$ of every direction
  and the effective number of parameters;
- set up weight decay in a framework optimizer.

**Running example:** 200 features, 20 training examples,
$\mathbf{w}^* = (0.01, \ldots, 0.01)$, $b^* = 0.05$, noise
$\sigma = 0.01$.
:::

::: {.slide title="Weight Decay Controls Capacity Continuously"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Linear Regression | model, loss, closed form, minibatch SGD |
| Object-Oriented Design | `Module`, `DataModule`, `Trainer` |
| Synthetic Data | data with known $\mathbf{w}^*, b^*$ |
| Implementation from Scratch | SGD and the training loop by hand |
| Concise Implementation | framework layers and losses |
| Generalization | training vs. test error |
| **Weight Decay** | **a penalty on large weights** |
:::

::: {.col .narrow}
- **Uses:** normal equations, eigenvectors of a symmetric matrix,
  Gaussian noise and maximum likelihood, validation sets.
- **Skips:** lasso solvers, the proof that penalty and constraint agree,
  and weight decay with adaptive optimizers.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The Penalty]{.dtitle}

[the loss · the update · the geometry · the prior]{.dsub}
:::
:::

::: {.slide title="The Penalty Prefers the Smallest of Equally Good Fits"}
[The penalty]{.kicker}

$$L(\mathbf{w}, b) + \frac{\lambda}{2}\,\|\mathbf{w}\|^2, \qquad L = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\big(\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)}\big)^2$$

- $\lambda = 0$: plain loss. Large $\lambda$: $\mathbf{w} \to \mathbf{0}$.
  Choose $\lambda$ on validation data.
- The bias is not penalized.
- Why small weights: by Cauchy–Schwarz,
  $|\mathbf{w}^\top\mathbf{x} - \mathbf{w}^\top\mathbf{x}'| \leq \|\mathbf{w}\|\,\|\mathbf{x} - \mathbf{x}'\|$.
  Norms: $(1, 1) \to 1.4$, $(501, -499) \to 707$.

::: {.d2l-note .warn}
This bound needs a linear model, whose weights are its slopes.
:::
:::

::: {.slide title="Each SGD Step Multiplies the Weights by 1 − ηλ"}
[The penalty]{.kicker}

The gradient of $\frac{\lambda}{2}\|\mathbf{w}\|^2$ is $\lambda\mathbf{w}$:

$$\mathbf{w} \leftarrow (1 - \eta\lambda)\,\mathbf{w} - \frac{\eta}{|\mathcal{B}|}\sum_{i \in \mathcal{B}_t} \mathbf{x}^{(i)}\big(\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)}\big)$$

- Without data the weights **decay**: $(1 - \eta\lambda)^t$ after $t$
  steps.
- $\eta = 0.01$, $\lambda = 3$: 0.97 per step, $0.97^{120} \approx 0.026$.

::: {.d2l-note}
Under SGD, penalty in the loss = shrink factor in the step. Frameworks
apply the decay inside the optimizer, not in the loss.
:::
:::

::: {.slide title="Ridge Shrinks, Lasso Selects"}
[The penalty]{.kicker}

Penalty ⇔ constraint: the smallest loss ellipse around $\hat{\mathbf{w}}$
touches the $\ell_2$ ball (left) or the $\ell_1$ diamond (right) at
$\hat{\mathbf{w}}_\lambda$.

![](../img/mdl-linreg-ridge-geometry.svg){width=80%}

Lasso ($\lambda\|\mathbf{w}\|_1$): no preference between $(1, 1)$ and
$(2, 0)$; ridge prefers $(1, 1)$.
:::

::: {.slide title="Weight Decay Is MAP Estimation With a Gaussian Prior"}
[The penalty]{.kicker}

Noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ gives the squared loss;
add a prior $w_j \sim \mathcal{N}(0, \tau^2)$:

$$-\log p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) = \frac{1}{2\sigma^2}\sum_{i=1}^n \big(y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)} - b\big)^2 + \frac{1}{2\tau^2}\|\mathbf{w}\|^2 + \textrm{const}$$

Multiply by $\sigma^2/n$:

$$L(\mathbf{w}, b) + \frac{\lambda}{2}\|\mathbf{w}\|^2 \quad \textrm{with} \quad \lambda = \frac{\sigma^2}{n\tau^2}$$

::: {.d2l-note}
Stronger penalty for noisier labels and narrower priors; weaker as $n$
grows.
:::
:::

::: {.slide title="Check: Which λ Does the Prior Suggest?"}
[Check]{.kicker}

Label noise $\sigma = 0.1$, prior $\tau = 1$, and $n = 100$ examples.
What is $\lambda$? What does it become with $10{,}000$ examples, and with
labels ten times noisier?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\lambda = 0.01/(100 \cdot 1) = 10^{-4}$. With $10{,}000$
examples it is $10^{-6}$: the data outweigh the prior. With
$\sigma = 1$ it is $10^{-2}$, a hundred times stronger.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[The Experiment]{.dtitle}

[200 features · 20 examples · with and without the penalty]{.dsub}
:::
:::

::: {.slide title="200 Weights and 20 Examples Leave Many Exact Fits"}
[The experiment]{.kicker}

$y = 0.05 + \sum_{j=1}^{200} 0.01\,x_j + \epsilon$,
$\epsilon \sim \mathcal{N}(0, 0.01^2)$; 20 training and 1,000
validation examples, minibatches of 5.

@weight-decay-defining-ell-2-norm-penalty

`WeightDecayScratch.training_step` returns
`loss + self.lambd * l2_penalty(self.w)`.

::: {.d2l-note}
Training minimizes the penalized loss; validation reports the plain
squared loss, comparable across $\lambda$.
:::
:::

::: {.slide title="Without the Penalty the Validation Loss Stays Near 0.02"}
[The experiment]{.kicker}

::: {.cols .vc}
::: {.col}
- $\lambda = 0$, 30 epochs, $\eta = 0.01$: 120 steps.
- Training loss below $10^{-7}$: all 20 labels fit.
- Validation loss stays near **0.02** (0.015 to 0.025 between runs).
- Norm of $\mathbf{w}$: 0.12 to 0.16.
:::

::: {.col .fig .big}
@!weight-decay-training-without-regularization
:::
:::
:::

::: {.slide title="λ = 3 Halves the Validation Loss"}
[The experiment]{.kicker}

::: {.cols .vc}
::: {.col}
- The training curve shows the penalized loss, about $10^{-3}$ to
  $10^{-2}$.
- Validation loss falls to about **0.01**.
- Norm of $\mathbf{w}$: 0.02 to 0.06 (0.12 to 0.16 at $\lambda = 0$).
:::

::: {.col .fig .big}
@!weight-decay-using-weight-decay
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Why It Works]{.dtitle}

[shrinkage along the eigenvectors of XᵀX/n]{.dsub}
:::
:::

::: {.slide title="Ridge Multiplies Each Eigen-Coordinate by μ/(μ + λ)"}
[Why it works]{.kicker}

No bias (or centered data). $\frac{1}{n}\mathbf{X}^\top\mathbf{X}$ has eigenvectors
$\mathbf{v}_j$, eigenvalues $\mu_j = \frac{1}{n}\|\mathbf{X}\mathbf{v}_j\|^2$.
Write $\mathbf{w} = \sum_j c_j\mathbf{v}_j$ and
$\frac{1}{n}\mathbf{X}^\top\mathbf{y} = \sum_j g_j\mathbf{v}_j$:

$$\Big(\tfrac{1}{n}\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}\Big)\mathbf{w} = \tfrac{1}{n}\mathbf{X}^\top\mathbf{y} \;\Longrightarrow\; c_j = \frac{\mu_j}{\mu_j + \lambda}\cdot\frac{g_j}{\mu_j}$$

- $\mu_j \gg \lambda$: kept. $\mu_j \ll \lambda$: suppressed.
  $\mu_j = 0$: set to 0.
- $\textrm{df}(\lambda) = \sum_j \mu_j/(\mu_j + \lambda)$: the effective
  number of parameters.

::: {.d2l-note}
Small $\mu_j$ is where label noise is amplified most, by $1/\mu_j$.
:::
:::

::: {.slide title="A Direction Is Kept While λ Is Below Its Eigenvalue"}
[Why it works]{.kicker}

![Shrinkage factor μ/(μ + λ) against λ for μ = 0.01, 1, 100; it is 1/2 at λ = μ. Directions with μ = 0 are set to zero.](../img/mdl-linreg-ridge-shrinkage.svg){width=60%}

Duplicated features: the sum direction has $\mu$ between 1 and 3, the
difference $\mu < 10^{-6}$.
:::

::: {.slide title="Least Squares Splits the Sum at Random; Ridge Returns (1, 1)"}
[Why it works]{.kicker}

$x_2 = x_1 + 0.001\,z$, $y = x_1 + x_2 + \epsilon$,
$\sigma = 0.1$, $n = 100$, $\lambda = 0.01$; tests with
$x_2 = x_1 + 0.001\,z$ and with $x_2 = x_1 + 0.1\,z$:

@!weight-decay-two-duplicated-features-2

- Least-squares weights always sum to about 2; their split varies widely.
- Disagreeing features: error $\approx 0.01\,(w_1 - 1)^2 + 0.01$ for least
  squares, about 0.01 for ridge.
:::

::: {.slide title="Twenty Examples Determine Only 20 of 200 Directions"}
[Why it works]{.kicker}

@!weight-decay-why-shrinkage-helps-the-spectral-view

- 180 eigenvalues are zero up to rounding.
- At $\lambda = 3$ the 20 others keep 60–85% of their coordinates.
- **About 15 effective parameters** out of 200 weights.
:::

::: {.slide title="Without Decay the Undetermined Part Keeps Its Initial Value"}
[Why it works]{.kicker}

Gradients $\frac{1}{n}\mathbf{X}^\top\mathbf{r}$ combine training inputs,
so they never move $\mathbf{w}$ along $\mathbf{X}\mathbf{v} = \mathbf{0}$:

@!weight-decay-the-directions-our-data-determine-1

- $\lambda = 0$: rest $\approx 0.01\sqrt{180} \approx 0.13$, the
  initialization. It adds $0.13^2/2 \approx 0.009$ to the validation loss.
- $\lambda = 3$: $0.97^{120} \approx 0.026$ of it remains.
:::

::: {.slide title="Check: What If We Start From Zero?"}
[Check]{.kicker}

Train both models from $\mathbf{w} = \mathbf{0}$ instead. What happens to
the validation loss at $\lambda = 0$? Is weight decay then useless?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The undetermined part stays 0, so $\lambda = 0$ also reaches
about 0.01, as with $\lambda = 3$. Here the gain came from removing the
initialization. With
correlated features, weight decay still damps the directions with small
$\mu_j$, which no starting point fixes.
:::
:::
:::

::: {.slide title="PyTorch Decays One Parameter Group" only="pytorch"}
[Concise]{.kicker}

`WeightDecay` extends the concise `LinearRegression` and replaces only its
optimizer:

```{.python #weight-decay-exercises-1}
def configure_optimizers(self):
    return torch.optim.SGD([
        {'params': self.net.weight, 'weight_decay': self.wd},
        {'params': self.net.bias}], lr=self.lr)
```

`weight_decay` adds `weight_decay` · $\mathbf{w}$ to the gradient of the
weight group; the bias group has none.
:::

::: {.slide title="Gluon Decays Every Parameter With wd_mult > 0" only="mxnet"}
[Concise]{.kicker}

`WeightDecay` extends the concise `LinearRegression` and replaces only its
optimizer:

```{.python #weight-decay-exercises-2}
def configure_optimizers(self):
    for p in self.collect_params('.*bias').values():
        p.wd_mult = 0
    return gluon.Trainer(self.collect_params(), 'sgd',
                         {'learning_rate': self.lr, 'wd': self.wd})
```

`wd` adds `wd` · $\mathbf{w}$ to the gradient, times `wd_mult`; the bias
gets `wd_mult = 0`.
:::

::: {.slide title="Keras Shrinks Every Variable Except the Bias" only="tensorflow"}
[Concise]{.kicker}

`WeightDecay` extends the concise `LinearRegression` and replaces only its
optimizer:

```{.python #weight-decay-exercises-3}
def configure_optimizers(self):
    optimizer = tf.keras.optimizers.SGD(self.lr, weight_decay=self.wd)
    optimizer.exclude_from_weight_decay(var_names=['bias'])
    return optimizer
```

`weight_decay` multiplies each variable by $1 - \eta\,\cdot$
`weight_decay` before the step; for SGD that equals adding it to the
gradient.
:::

::: {.slide title="Optax Chains a Masked Decay Before SGD" only="jax"}
[Concise]{.kicker}

`WeightDecay` extends the concise `LinearRegression` and replaces only its
optimizer:

```{.python #weight-decay-exercises-4}
def is_matrix(params):
    return jax.tree.map(lambda p: p.ndim > 1, params)

def configure_optimizers(self):
    return optax.chain(
        optax.masked(optax.add_decayed_weights(self.wd), is_matrix),
        optax.sgd(self.lr))
```

`add_decayed_weights` adds `wd` · $\mathbf{w}$ to the gradient; `masked`
restricts it to parameters with more than one dimension (the kernel).
:::

::: {.slide title="Halve η, Double the Decay, and the Steps Match"}
[Concise]{.kicker}

::: {.cols .vc}
::: {.col}
Library losses omit the $\frac{1}{2}$: their gradient is twice ours.

- `lr=0.005` restores our data step at $\eta = 0.01$.
- The decay $\lambda\mathbf{w}$ is not doubled, so `wd=6`:
  $1 - 0.005 \cdot 6 = 0.97$.

::: {.d2l-note}
Norm 0.02 to 0.06, as from scratch at $\lambda = 3$; validation MSE near
0.02, which is 0.01 in our units.
:::
:::

::: {.col .fig .big}
@!weight-decay-concise-implementation-2
:::
:::
:::

::: {.slide title="Check: Which Scratch Update Does a Framework Run?"}
[Check]{.kicker}

A framework trains on the mean squared error (no $\frac{1}{2}$) with
learning rate 0.1 and weight decay $10^{-4}$. Which $\eta$ and $\lambda$
of our update
$\mathbf{w} \leftarrow (1 - \eta\lambda)\mathbf{w} - \eta\nabla L$ does
it run?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Its step is
$\mathbf{w} - 0.1\,(2\nabla L + 10^{-4}\mathbf{w}) = \mathbf{w} - 0.2\,(\nabla L + 5 \cdot 10^{-5}\mathbf{w})$:
$\eta = 0.2$ and $\lambda = 5 \cdot 10^{-5}$.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Penalty:** $L + \frac{\lambda}{2}\|\mathbf{w}\|^2$; among equal fits,
  the smallest weights win; $\lambda$ chosen on validation data.
- **Update:** multiply by $1 - \eta\lambda$ each step (0.97 here).
- **Lasso** selects (corners); **ridge** shrinks (round ball).
- **MAP:** Gaussian prior, $\lambda = \sigma^2/(n\tau^2)$.
- **Shrinkage:** $\mu_j/(\mu_j + \lambda)$ per eigenvector. For duplicated
  features, ridge returns about $(1, 1)$ where least squares splits the
  sum at random. Our data: 20 of 200 directions determined,
  $\textrm{df}(3) \approx 15$.
- **Experiment:** validation loss 0.02 → 0.01; the decay removed the
  initialization from the 180 undetermined directions (a start from zero
  does as well).

**Next:** classification, with the same `Module`, `DataModule`, and
`Trainer`.
:::
