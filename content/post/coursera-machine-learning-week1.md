---
title: "Machine Learning, Week 1"
date: 2017-08-31T10:25:51-06:00
tags: ["Machine Learning"]
draft: false
---

Taking the [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/).

{{% toc %}}

* Lecture notes:
  * [Lecture1](/docs/coursera-machine-learning-week1/Lecture1.pdf)
  * [Lecture2](/docs/coursera-machine-learning-week1/Lecture2.pdf)
  * [Lecture3](/docs/coursera-machine-learning-week1/Lecture3.pdf)

# Introduction

## Machine Learning

### What is Machine Learning

* **Arthur Samuel (1959)**: The field of study that gives computers the ability to
learn without explicitly programmed.
* **Tom Mitchell (1998)**: Well-posed Learning Problem; A computer program is said to _learn_ from experience **E** with respect to some task **T** and some performance measure **P** if its performance on **T**, as measured by **P**, improves with experience **E**.

Example: playing checkers.

* **E** = The experience of playing many games of checkers.
* **T** = The task of playing checkers.
* **P** = The probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications, Supervised learning and Unsupervised learning.

### Supervised Learning

In supervised learning, we have a data set and we already know what the correct output should look like. There is an idea that a relationship exists between the input and output.

Supervised learning is categorized into **regression** and **classification** problems.

* **Regression**: Results are within a continuous output. We are trying to map input variables to some continuous function.
* **Classification**: Results are discrete. We are trying to map input variables into separate categories.

Example 1:

Given data about the sizes of houses on the real estate market, attempt to predict price. Price as a function of size is a continuous output, so this is a _regression_ problem.

Example 2:

Given data about a patient with a tumor, predict whether or not the tumor is malignant or benign. The function does not produce a continuous output, only two categories are given, therefore this is a _classification_ problem.

Examples:

* Given email labeled as spam/not spam, learn a spam filter
* Given a dataset of patients diagnosed as either having diabetes or not, learn to classify new patients as having diabetes or not.

### Unsupervised Learning

Unsupervised learning allows approaches to problems with little or no idea what the results should look like. Structure is derived from data where we do not know the effect of the variables. This can be done by clustering the data based on relationships or variables within the data.

With unsupervised learning, there is no feedback based on the prediction results.

Examples:

* **Clustering**: Take a collection fo 1,000,000 different genes and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, etc.
* **Non-Clustering**: The "Cocktail Party Algorithm" allows you to find structure in a chaotic environment (such as identifying individual voices and music from a mesh of sounds at a cocktail party).

* Given a set of news articles found on the web, group them into sets of articles about the same stories.
* Given a database of customer data, automatically discover market segments and group customers into different market segments.

## Linear Regression with One Variable

### Model Representation

This is the notation we will use moving forward.

* Input variables (features) are denoted as $x^{(i)}$.
* Output variables are denoted as $y^{(i)}$.
* A pair $(x^{(i)}, y^{(i)})$ is called a training example.
* The dataset we'll be using to learn is a list of training examples is called a training set and is denoted as $(x^{(i)}, y^{(i)}); i = 1, ..., m$
* The value $X$ is used to denote the space of input values and $Y$ is used to denote the space of output values.
  * $X = Y = \mathbb{R}$
* Note: The _(i)_ is not exponentiation, but identifying.

Given a training set, learn a function $h:X \rightarrow Y$ such that $h(x)$ is a _good_ predictor for the corresponding value of $y$. For historical reasons, the function $h$ is called a hypothesis.

![hypothesis](/img/coursera-machine-learning-week1/hypothesis.png)

### Cost Function & Intuitions

We measure the accuracy of a hypothesis functions by using a **cost function**. This takes an average difference of all the results of the hypothesis with inputs from X and the outputs Y.

The cost function we will be using for now is the **Squared Error Function**, also known as **Mean Squared Error**.

$$ J(\theta\_{0}, \theta\_{1}) = \dfrac{1}{2m} \sum\_{i=1}^m (\hat{y}\_{i} - y\_{i})^{2} = \dfrac{1}{2m} \sum\_{i=1}^m (h\_{\theta}(x\_{i}) - y\_{i})^{2}  $$

Thinking about this in visual terms, training data set is scattered on the x,y plane. We are trying to make a straight line pass through these scattered points. We want the best possible line such that the average squared vertical distances of the scattered points from the line will be the least.

![cost_function_1](/img/coursera-machine-learning-week1/cost_function_1.png)
![cost_function_2](/img/coursera-machine-learning-week1/cost_function_2.png)
![cost_function_3](/img/coursera-machine-learning-week1/cost_function_3.png)
![cost_function_4](/img/coursera-machine-learning-week1/cost_function_4.png)

### Gradient Descent

Gradient descent is a method of estimating the parameters in the hypothesis function using the cost function. Imagine that we graph the hypothesis function based on its fields $\theta\_0, \theta\_1$. We put these variables on the x and y axis and we plot the cost function on the vertical z axis. The points on the graph will be the result of the cost function using the hypothesis with those specific theta parameters.

![gradient_descent_1](/img/coursera-machine-learning-week1/gradient_descent_1.png)

We need to minimize our cost function by "stepping" down from the top to the bottom points of this graph. The red arrows show local minimums in the graph.

This is done by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter (alpha), which is the learning rate.

The gradient descent algorithm is:

_repeat until convergence:_
$$\theta\_j := \theta\_j - \alpha \dfrac{\partial}{\partial\theta\_j} J(\theta\_0,\theta\_1)$$
_where:_ $j = 0,1$ represents the feature index number.

At each iteration j, one should simultaneously update the parameters $\theta\_1, \theta\_2, \dots, \theta\_n$. Updating a specific parameter prior to calculating another one on the $j^{(th)}$ iteration leads to a wrong implementation.

![gradient_descent_2](/img/coursera-machine-learning-week1/gradient_descent_2.png)

Regardless of the slope's sign for the derivative, $\theta\_1$ eventually converges to its minimum value. The following figure shows that when the slope is negative, the value of $\theta\_1$ increases. When the slope is positive, the value of $\theta\_1$ decreases.

![gradient_descent_3](/img/coursera-machine-learning-week1/gradient_descent_3.png)

We must adjust the parameter alpha to ensure that the gradient descent algorithm converges in reasonable time. Failure to converge or too much time to obtain the minimum value implies that the step size is wrong.

![gradient_descent_4](/img/coursera-machine-learning-week1/gradient_descent_4.png)

Even with a fixed step size, gradient descent can converge. The reason is because as we approach the bottom of the convex function, the derivative approaches zero.

![gradient_descent_5](/img/coursera-machine-learning-week1/gradient_descent_5.png)

### Gradient Descent for Linear Regression

Applying the gradient descent algorithm to the cost functions defined earlier, we must calculate the necessary derivatives.

![gradient_descent_linear_regression_1](/img/coursera-machine-learning-week1/gradient_descent_linear_regression_1.png)

This gives us the new gradient descent algorithm:

_repeat until convergence:_
$$\theta\_0 := \theta\_0 - \alpha \frac{1}{m} \sum\limits\_{i=1}^{m}(h\_\theta(x\_{i}) - y\_{i})$$
$$\theta\_1 := \theta\_1 - \alpha \frac{1}{m} \sum\limits\_{i=1}^{m}((h\_\theta(x\_{i}) - y\_{i}) x\_{i}$$

If we start with a guess for our hypothesis function and we repeatedly apply the gradient descent equations, our hypothesis will become more and more accurate.

This is simply gradient descent on the original cost function J. This method looks at every example in the entire training set at every step, therefore this is called **batch gradient descent**. Note: while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima. Thus, gradient descent here always converges (assuming alpha isn't too large) to the global minimum. J is a convex quadratic function.

![gradient_descent_linear_regression_2](/img/coursera-machine-learning-week1/gradient_descent_linear_regression_2.png)

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48, 30). The x's in the figure represent each step in gradient descent as it converged to its minimum.

# Optional Linear Algebra

## Linear Algebra Review

### Matrices and Vectors

Matrices are 2-dimensional arrays:

$$\begin{bmatrix} a & b & c \newline d & e & f \newline g & h & i \newline j & k & l\end{bmatrix}$$

The above matrix has four rows and three columns, therefore it is a 4 x 3 matrix.

A vector is a matrix with one column and many rows:

$$\begin{bmatrix} w \newline x \newline y \newline z \end{bmatrix}$$

Vectors are a subset of matrices. The above vector is a 4 x 1 matrix.

**Notation and terms:**

* $A\_{ij}$ refers to the element in the ith row and jth column of matrix A.
* A vector with 'n' rows is referred to as an 'n'-dimensional vector.
* $v\_i$ refers to the element in the ith row of the vector.
* In general, all vectors and matrices will be 1-indexed from the top left corner.
* Matrices are usually denoted by uppercase names while vectors are lowercase.
* "Scalar" means that an object is a single value, not a vector or matrix.
* $\mathbb{R}$ refers to the set of scalar real numbers.
* $\mathbb{R}^\mathbb{n}$ refers to the set of n-dimensional vectors of real numbers.

```octave
% The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

% Initialize a vector 
v = [1;2;3] 

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Get the dimension of the vector v 
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)
```
```text
A =

    1    2    3
    4    5    6
    7    8    9
   10   11   12

v =

   1
   2
   3

m =  4
n =  3
dim_A =

   4   3

dim_v =

   3   1

A_23 =  6
```

### Matrix Addition and Scalar Operations

Addition and Subtraction occur **element-wise**, simply add or subtract each corresponding element.

$$\begin{bmatrix} a & b \newline c & d \newline \end{bmatrix} +\begin{bmatrix} w & x \newline y & z \newline \end{bmatrix} =\begin{bmatrix} a+w & b+x \newline c+y & d+z \newline \end{bmatrix}$$
$$\begin{bmatrix} a & b \newline c & d \newline \end{bmatrix} - \begin{bmatrix} w & x \newline y & z \newline \end{bmatrix} =\begin{bmatrix} a-w & b-x \newline c-y & d-z \newline \end{bmatrix}$$

To add or subtract two matrices, their dimensions must be **the same**.

In scalar multiplication and division, we simply multiply or divide every element by the scalar value.

$$\begin{bmatrix} a & b \newline c & d \newline \end{bmatrix} * x =\begin{bmatrix} a*x & b*x \newline c*x & d*x \newline \end{bmatrix}$$
$$\begin{bmatrix} a & b \newline c & d \newline \end{bmatrix} / x =\begin{bmatrix} a /x & b/x \newline c /x & d /x \newline \end{bmatrix}$$

```octave
% Initialize matrix A and B 
A = [1, 2, 4; 5, 3, 2]

% Initialize constant s 
s = 2

% What happens if we have a Matrix + scalar?
add_As = A + s

sub_As = A - s
```

```text
A =

   1   2   4
   5   3   2

s =  2
add_As =

   3   4   6
   7   5   4

sub_As =

  -1   0   2
   3   1   0
```

### Matrix-Vector Multiplication

When multiplying a matrix with a vector, we map the column of the vector onto each row of the matrix, multiplying each element and summing the result.

$$\begin{bmatrix} a & b \newline c & d \newline e & f \end{bmatrix} *\begin{bmatrix} x \newline y \newline \end{bmatrix} =\begin{bmatrix} a*x + b*y \newline c*x + d*y \newline e*x + f*y\end{bmatrix}$$

An **m x n matrix** multiplied by an **n x 1 vector** results in an **m x 1 vector**.

The result is a **vector**. The number of **columns of the matrix** must equal the number of **rows in the vector**.

```octave
% Initialize matrix A 
A = [1, 2, 3; 4, 5, 6;7, 8, 9; 434, 54, 3] 

% Initialize vector v 
v = [1; 1; 1;] 

% Multiply A * v
Av = A * v
```
```text
A =

     1     2     3
     4     5     6
     7     8     9
   434    54     3

v =

   1
   1
   1

Av =

     6
    15
    24
   491
```

### Matrix-Matrix Multiplication

Two matrices are multiplied by breaking it into several vector multiplications and concatenating the result.

$$\begin{bmatrix} a & b \newline c & d \newline e & f \end{bmatrix} *\begin{bmatrix} w & x \newline y & z \newline \end{bmatrix} =\begin{bmatrix} a*w + b*y & a*x + b*z \newline c*w + d*y & c*x + d*z \newline e*w + f*y & e*x + f*z\end{bmatrix}$$

An **m x n matrix** multiplied by an **n x o matrix** results in an **m x o matrix**. In the above example, a 3 x 2 matrix multiplied by a 2 x 2 matrix resulted in a 3 x 2 matrix.

![matrix_matrix_multiplication](/img/coursera-machine-learning-week1/matrix_matrix_multiplication.png)

To multiply two matrices, the number of **columns** of the first matrix must equal the number of **rows** of the second matrix.

```octave
% Initialize a 3 by 2 matrix 
A = [1, 2; 3, 4;5, 6]

% Initialize a 2 by 1 matrix 
B = [1; 2] 

% We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1) 
mult_AB = A*B
```
```text
A =

   1   2
   3   4
   5   6

B =

   1
   2

mult_AB =

    5
   11
   17
```

### Matrix Multiplication Properties

Matrices are **not commutative**, that is $A * B \neq B * A$. Matrices are **associative**, that is $(A * B) * C = A * (B * C)$

Identity Matrices, when multiplied by any matrix of the same dimensions, returns the original matrix. Identity matrices have ones along the diagonals and zeros everywhere else. They are **n x n** dimensioned.

$$\begin{bmatrix} 1 & 0 & 0 \newline 0 & 1 & 0 \newline 0 & 0 & 1 \newline \end{bmatrix}$$

![identity_matrix](/img/coursera-machine-learning-week1/identity_matrix.png)

Note, the identity matrix $I$ does not have the same dimensions in $A * I$ and $I * A$, as the dimensions of $I$ are implicit from the context of $A$.

```octave
% Initialize random matrices A and B 
A = [1,2;4,5]
B = [1,1;0,2]

% Initialize a 2 by 2 identity matrix
I = eye(2)

% The above notation is the same as I = [1,0;0,1]

% What happens when we multiply I*A ? 
IA = I*A 

% How about A*I ? 
AI = A*I 

% Compute A*B 
AB = A*B 

% Is it equal to B*A? 
BA = B*A 

% Note that IA = AI but AB != BA
```
```text
A =

   1   2
   4   5

B =

   1   1
   0   2

I =

Diagonal Matrix

   1   0
   0   1

IA =

   1   2
   4   5

AI =

   1   2
   4   5

AB =

    1    5
    4   14

BA =

    5    7
    8   10
```

### Inverse and Transpose

When a matrix is multiplied by its inverse, you get the identity. The **inverse** of a matrix $A$ is denoted $A^{-1}$.

$$ A * A^{-1} = I $$

A non square matrix does not have an inverse matrix. We can compute inverses of matrices in octave with the `pinv(A)` function, and in Matlab with the `inv(A)` function. Matrices that do not have an inverse are _singular_ or _degenerate_.

The **transposition** of a matrix is like the result of rotating the matrix $90^\circ$ in a clockwise direction then reversing it. This can be computed in octave with `A'` or in Matlab with the `transpose(A)` function.

$$A = \begin{bmatrix} a & b \newline c & d \newline e & f \end{bmatrix}$$
$$A^T = \begin{bmatrix} a & c & e \newline b & d & f \newline \end{bmatrix}$$

In other words: $A\_{ij} = A^T\_{ji}$

```octave
% Initialize matrix A 
A = [1,2,0;0,5,6;7,0,9]

% Transpose A 
A_trans = A' 

% Take the inverse of A 
A_inv = inv(A)

% What is A^(-1)*A? 
A_invA = inv(A)*A
```
```text
A =

   1   2   0
   0   5   6
   7   0   9

A_trans =

   1   0   7
   2   5   0
   0   6   9

A_inv =

   0.348837  -0.139535   0.093023
   0.325581   0.069767  -0.046512
  -0.271318   0.108527   0.038760

A_invA =

   1.00000  -0.00000   0.00000
   0.00000   1.00000  -0.00000
  -0.00000   0.00000   1.00000
```

---

Move on to [Week 2]({{% relref "coursera-machine-learning-week2.md" %}}).
