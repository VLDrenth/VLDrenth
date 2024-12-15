---
title: "Beyond LASSO"
description: "A brief exploration of LASSO and its variants"
date: 2024-12-15
draft: false
math: true
tags: ["shrinkage", "lasso"]
customJS:
  - /js/lasso-visualization.js
---


# The basics
One of the first 'machine learning' many will come across is LASSO regression. This may suggest that it's an elementary method; something to get you started on machine learning before moving to the more interesting topics. But there is a sea of methods that are adaptions of LASSO. The goal of this blog post is to give a brief overview of the landscape of these methods and where they might be useful. Understanding these other methods, many of which generalize the LASSO, should also help us gain a better understanding (and appreciation) of the 'simple' version. 

Let's first review the basic idea behind LASSO. We start with the usual linear regression setup:
$$
y = X\beta + \varepsilon.
$$

If the number of features is not too high, the OLS estimator gives good estimates of the coefficients and all is well. However, when the number of features increases, this increases the total variance of the OLS estimates and may lead to worse predictions out of sample. Estimates for the effect of a feature are more likely to be very large, which doesn't tend to generalize well. If we have a lot of features, possibly more than the number of observations, the OLS estimator is not even defined as there is no unique solution to the least-squares problem. One way to circumvent this problem is to simply throw away some of your features. This can be done using domain knowledge, some data analysis, or some automated method. This may feel rather ad-hoc however. An alternative, more principled way, is to use _regularization_. The key idea is that large coefficient estimates may overestimate the _true_ effect size, so we should simply penalize the size of the coefficient! This simple but powerful idea is the driving idea behind the large literature on shrinkage methods.

# Ridge Regression
The simplest way to regularize the coefficient size is to add a penalty on the size of the coefficients. A natural way to capture the size of $\beta$ is to use the Euclidean norm:

$$
\min_{\boldsymbol{\beta}} \left\{ \|\boldsymbol{y} - \boldsymbol{X\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_2^2 \right\}
$$

This is called Ridge regression, which has a unique minimizer for any positive penalty strengthThis is called Ridge regression, which has a unique minimizer for any positive penalty strength $$\lambda$$, even when we have many more features than observations! We can even get the solution in closed form by making use of the fact that the Euclidean norm is differentiable. This is a natural and convenient choice, but the Euclidean norm is only one out of many. Why not use the $$L_1$$ norm? This is also a viable choice, and is what gives us the LASSO estimator.

# LASSO Regression
LASSO shrinkage simply applies a different norm as penalty, but there are many differences. Just to spell it out, this is the problem LASSO solves: 
$$
\min_{\boldsymbol{\beta}} \left\{\|\boldsymbol{y} - \boldsymbol{X\beta}\|_2^2 + \lambda\sum_{j=1}^p|\boldsymbol{\beta}_j| \right\}.
$$
While similar to Ridge, there are many key differences to explore. First a practical issue: the $$L_1$$ norm is not differentiable, so there is no closed form solution anymore. Thankfully, it is true that all valid norms are convex (how lucky!), which means this is still an 'easy' problem to solve. More on that later.

More interesting, is how the two penalties differ in the resulting estimates. To explain this, I can't do any better than the bible of classical ML: the Elements of Statistical Learning. The gist of the argument is this. We can consider the minimization problems above as Lagrangians of a constrained optimization problem. Both Ridge and LASSO aim to minimize the sum of squared errors, but differ in their constraint. The Ridge constraint is $\| \boldsymbol{\beta} \|_2^2 \leq t$ and analously the LASSO constraint is $\| \boldsymbol{\beta} \|_1 \leq t$ (Exercise: What is the value of $t$?). Below you see the feasible region in a simple 2D case as well as the contours which represent a value of the squared loss for some set of coefficients. Notice that the LASSO region has a sharp edge and may intersect with a contour line at a corner, where one of the coefficients equals zero. Ridge, on the other hand, is smooth and will always have non-zero coefficients. This is the crucial difference between Ridge and LASSO: LASSO sets some subset of the predictor's coefficients to zero and thus searches for a _sparse_ solution. 

<div style="display: flex; justify-content: center; gap: 20px; margin: 20px 0;">
  <div>
    <h4 style="text-align: center;">L1 (LASSO)</h4>
    <canvas id="l1Canvas" width="300" height="300" style="border:1px solid #000;"></canvas>
  </div>
  <div>
    <h4 style="text-align: center;">L2 (Ridge)</h4>
    <canvas id="l2Canvas" width="300" height="300" style="border:1px solid #000;"></canvas>
  </div>
</div>
<div style="text-align: center; margin: 10px 0;">
  <label for="tSlider">Constraint bound (t): </label>
  <input type="range" id="tSlider" min="0.1" max="2" step="0.1" value="1" style="width: 200px;">
  <span id="tValue">1.0</span>
</div>

## The math behind LASSO
While this seems intuitive, let's make this a bit more precise. Recall that the LASSO objective is a convex optimization problem but not differentiable. While a differentiable (and convex) objective may be solved by setting the gradient to zero, here we need to rely on the _subgradient_, a generalized notion of the derivative. This gives us the condition for optimal $\beta$ as:

$$
X^T(Y - X\hat{\beta}) = \lambda s
$$

where $$
s_j = \begin{cases} 
-1, & \text{if } \beta_j < 0, \\
1 & \text{if } \beta_j > 0 \\
[-1, 1] & \text{if } \beta_j = 0
\end{cases}
$$.

## Estimating LASSO
With this in mind we can look at an algorithm to solve for $\hat{\beta}$. Suppose we only have 1 regressor, and that we are minimizing $\frac{1}{2N}\sum_{i=1}^N(y_i - x_i\beta)^2 + \lambda|\beta|$. By taking the (sub)-derivative and setting this equal to zero we obtain the following solution which depends on the penalization strength:

$$
\hat{\beta} =
\begin{cases} 
    \frac{1}{N}\langle x, y \rangle - \lambda, & \text{if } \frac{1}{N}\langle x, y \rangle > \lambda, \\ 
    0, & \text{if } \frac{1}{N} |\langle z, y \rangle| \leq \lambda, \\ 
    \frac{1}{N}\langle x, y \rangle + \lambda, & \text{if } \frac{1}{N}\langle x, y \rangle < -\lambda.
\end{cases}
$$

This is often written using the _soft thresholding operator_ as $$\beta = \mathcal{S}_\lambda(\frac{1}{N}\langle x, y \rangle)$$ where 
$$\mathcal{S}_\lambda(u) = sign(u)max(0, |u| - \lambda)$$. First off, note that when $$\lambda$$ is set to zero we indeed recover the OLS estimator. But when it's not equal to zero, the LASSO sets the coefficient to zero when the covariance is too small. Even when the covariance is larger than the $$\lambda$$, it is shrunk towards zero. This clearly showcases how LASSO shrinks estimates towards zero and may set them to *exactly* zero.

This is for a single covariate, which is not really what we're here for, so let's extend this to many regressors. We can use _cyclical coordinate descent_, which iterates over all coefficients $$1, \dots, p$$. The algorithm then minimizes with respect to a single $$\beta_j$$ keeping all others fixed. Doing this while iterating over coefficients until convergence yields an esimtate of the multivariate LASSO. It turns out that by rewriting the problem a little bit, we can use the univariate solution to do this! The objective can be written as: 
$$
\frac{1}{2N} \sum_{i=1}^N \left( y_i - \sum_{k \neq j} x_{ik} \beta_k - x_{ij} \beta_j \right)^2 +
\lambda \sum_{k \neq j} |\beta_k| + \lambda |\beta_j|
$$. 
If we are only optimizing for $$\beta_j$$, this is a univariate regression of the 'partial residuals' $r_j = y - X_{-j}\beta_{-j}$ on $x_j$, which has an explicit solution. Updating $\hat{\beta}_j \leftarrow \mathcal{S}_\lambda(\frac{1}{N}\langle x_j, r_j \rangle)$ whilst cycling over all coordinates gives the LASSO estimates. It is interesting to note what happens when our predictors are all orthogonal. In this case we have $\langle x_j, r_j \rangle = \langle x_j, y \rangle$.
In other words, the estimate for each predictors is just the univariate solution, meaning the whole problem can be solved in closed form! So you are concerned about convergence and speed, you can always orthogonalize your features and get the solution explicitly.

The convergence properties of this algorithm and generally the class of coordinate descent methods are interesting and well-studied, but I'll just say this convergences under normal circumstances and leave the details for another day. In practice more sophisticated methods are used, that construct the optimal estimates not just for a single value of $\lambda$ but for a full sequence or _path_.

## Other Sparse Penalties
Most (if not all) of the above may already be familiar to you from an intro to machine learning class. The remainder of this post hopefully shows at least 1 new method that might be useful for whatever problem you're working on. I'll cover the following versions or generalizations of the LASSO:

- Elastic Net
- Adaptive LASSO
- Relaxed LASSO
- Group LASSO