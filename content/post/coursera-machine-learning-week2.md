---
title: "Machine Learning, Week 2"
date: 2017-08-31T14:05:35-06:00
tags: ["Machine Learning"]
draft: true
---

Taking the [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/).

Assumes you have knowledge of [Week 1]({{% relref "coursera-machine-learning-week1.md" %}}).

{{% toc %}}

* Lecture notes:
  * [Lecture4](/docs/coursera-machine-learning-week2/Lecture4.pdf)
  * [Lecture5](/docs/coursera-machine-learning-week2/Lecture5.pdf)

# Linear Regression with Multiple Variables

## Multivariate Linear Regression

### Multiple Features

Linear regression with multiple variables is known as **Multivariate Linear Regression**. We can now introduce the following notation for equations where we can have any number of input variables.

- $x\_j^{(i)} = \text{value of the feature } j \text{ in the } i^{th} \text{ training example}$
- $x^{(i)} = \text{the input (features) of the } i^{th} \text{ training example}$
- $m = \text{the number of training examples}$
- $n = \text{the number of features}$

![multiple_features_1](/img/coursera-machine-learning-week2/multiple_features_1.png)

The multivariable form of the hypothesis function accommodating these multiple features is as follows:

$$h\_\theta(x) = \theta\_0 + \theta\_1x\_1 + \theta\_2x\_2 + \theta\_3x\_3 + \dots +  + \theta\_nx\_n$$

![multiple_features_2](/img/coursera-machine-learning-week2/multiple_features_2.png)

The multivariate form of the hypothesis function can be concisely represented as theta transpose x.

$$h\_\theta(x) = \begin{bmatrix} \theta\_0 \hspace{1em} \theta\_1 \hspace{1em} \dots \hspace{1em} \theta\_n \end{bmatrix} \begin{bmatrix} x\_0 \newline x\_1 \newline \vdots \newline x\_n \end{bmatrix}= \theta^T x$$

### Gradient Descent for Multiple Variables

The following image compares the old gradient descent formula with one variable, to gradient descent with multiple variables.

![gradient_descent_multiple_variables](/img/coursera-machine-learning-week2/gradient_descent_multiple_variables.png)

Essentially, the new gradient descent equation is the same form, simply repeated for the *n* number of features:

_repeat until convergence:_

$\theta\_0 := \theta\_0 - \alpha \dfrac{1}{m} \sum\limits\_{i=1}^{m} (h\_\theta(x^{(i)})-y^{(i)}) \cdot x\_0^{(i)}$

$\theta\_1 := \theta\_1 - \alpha \dfrac{1}{m} \sum\limits\_{i=1}^{m} (h\_\theta(x^{(i)})-y^{(i)}) \cdot x\_1^{(i)}$

$\theta\_2 := \theta\_2 - \alpha \dfrac{1}{m} \sum\limits\_{i=1}^{m} (h\_\theta(x^{(i)})-y^{(i)}) \cdot x\_2^{(i)}$

$\vdots$

$\theta\_n := \theta\_n - \alpha \dfrac{1}{m} \sum\limits\_{i=1}^{m} (h\_\theta(x^{(i)})-y^{(i)}) \cdot x\_n^{(i)}$

Or, in other words:

_repeat until convergence:_ $\theta\_j := \theta\_j - \alpha \dfrac{1}{m} \sum\limits\_{i=1}^m (h\_\theta(x^{(i)})-y^{(i)}) \cdot x\_j^{(i)}$ _for_ $j := 0 \dots n$

### Gradient Descent in Practice - Feature Scaling & Mean Normalization

The idea is to make sure the inputs (features) are on a similar scale.

Given the example of determining the price of a house, we have two features that we care about. $x\_1 \text{ (size of the house)}$ and $x\_2 \text{ (number of bedrooms)}$. Rather than using the raw values directly, it can be more efficient to scale the features such that the values are similar.

![feature_scaling](/img/coursera-machine-learning-week2/feature_scaling.png)

When performing feature scaling, we want to get every feature into approximately a $ -1 \leq x\_i \leq 1$ range. This is approximate, the range of values should be approximately around one order of magnitude of $\pm1$. 

In addition to feature scaling, one other way to normalize the inputs is to perform mean normalization.

Replace $x\_i$ with $x\_i - \mu\_i$ to make features have approximately zero mean. (Do not apply this to $x\_0 = 1$).

![mean_normalization](/img/coursera-machine-learning-week2/mean_normalization.png)

Combining both feature scaling and mean normalization, one gets the following:

$$ x\_i := \dfrac{x\_i - \mu\_i}{s\_i} $$
$\mu\_i = \text{ average value of } x\_i \text{ and } s\_i = \text{ range (max-min) or standard deviation of } x\_i $
 

Example:

Suppose you are using a learning algorithm to estimate prices of houses in a city. One of our features, $x\_i$ captures the age of the house. In the training set, the houses have an age between 12 and 50 years, with an average age of 42 years. The following would be an adequate use as features, assuming feature scaling and mean normalization is applied:

$$x\_i = \dfrac{\text{age of house} - 42}{38}$$

### Gradient Descent in Practice - Learning Rate

The following is how to debug whether or not your gradient descent is working, as well as how to choose your learning rate $\alpha$.

It may be useful to plot the value of $J(\theta)$ as you run gradient descent. The value of $\min\limits\_\theta J(\theta)$ should decrease as you perform more iterations of gradient descent. If the value **$J(\theta)$ is increasing**, this is an indication that one should **use a smaller $\alpha$ learning rate**.

![learning_rate](/img/coursera-machine-learning-week2/learning_rate.png)

For sufficiently small $\theta$, $J(\theta)$ should always decrease on every iteration. However, if $\alpha$ is too small, gradient descent can be slow to converge.

To choose $\alpha$, try:
$$\dots, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, \dots$$

### Features and Polynomial Regression

Given a the problem of determining the price of a land given it's size, imagine we were given "frontage" and "depth" of the lot of land. Because we know that "area" of the house is more important than length/width, it might be more useful to use a calculated feature "area" instead.

$$x\_3 = x\_1 * x\_2$$
$$\text{area} = \text{frontage} * \text{depth}$$

It is important to choose the right hypothesis function to fit your data. Insight into the problem you are trying to solve is useful in this case. The hypothesis function should not be linear (a straight line) if this does not fit the data well.

We can change the behavior or curve of the hypothesis function by making it quadratic, cubic, or square root (or any other form, really). For example, if the hypothesis function is $h\_\theta(x) = \theta\_0 + \theta\_1x\_1$ then we can create additional features based on $x\_1$ to get the quadratic function $h\_\theta(x) = \theta\_0 + \theta\_1x\_1 + \theta\_2x\_1^2$ or the cubic function $h\_\theta(x) = \theta\_0 + \theta\_1x\_1 + \theta\_2x\_1^2 + \theta\_3x\_1^3$


In the cubic version, we created two new features: $x\_2 = x\_1^2 \text{,} \hspace{1em} x\_3 = x\_1^3$. 

One important thing to keep in mind is, if features are chosen this way then feature scaling becomes much more important.

$$\text{If } 1 \lt x\_1 \lt 1000 \text{ then } 1 \lt x\_1^2 \lt 1000000 \text{ then } 1 \lt x\_1^3 \lt 1000000000$$

Example:

Suppose you want to predict a house's price as a function of its size. Your model is

$$h\_\theta(x) = \theta\_0 + \theta\_1(\text{size}) + \theta\_2\sqrt{(\text{size})}$$

Suppose size ranges from $1 \text{ to } 1000 \text{(feet}^2\text{)}$. You will implement this by fitting a model

$$h\_\theta(x) = \theta\_0 + \theta\_1x\_1 + \theta\_2x\_2$$

Ignoring mean normalization, the following would be a valid choice for $x\_1$ and $x\_2$ ($\sqrt{1000} \approx 32$).

$$x\_1 = \dfrac{\text{size}}{1000} \text{,} \hspace{1em} x\_2 = \dfrac{\sqrt{\text{size}}}{32}$$

## Computing Parameters Analytically

### Normal Equation

Gradient descent gives one way to minimize $J$. Another way of doing this is to perform the minimization explicitly without resorting to using an interative algorithm. In the **Normal Equation** method, we will minimize $J$ by explicitly taking its derivatives with respect to the $\thetaJ$'s and setting them to zero. This allows us to compute the optimum theta without iteration.

We first must create $X$, which is also known as the _design matrix_. 
![normal_equation_1](/img/coursera-machine-learning-week2/normal_equation_1.png)

$$\theta = (X^TX)^{-1}X^Ty$$

![normal_equation_2](/img/coursera-machine-learning-week2/normal_equation_2.png)

The normal equation eliminates the need for feature scaling.

The following is a comparison of gradient descent and the normal equation:

| Gradient Descent           | Normal Equation              |
| -------------------------- | ---------------------------- |
| Con: Need to choose alpha  | Pro: No need to choose alpha |
| Con: Needs many iterations | Pro: No need to iterate      |
| Pro: $O(kn^2)$ complexity  | Con: $O(n^3)$ complexity     |
| Pro: Works well on large n | Con: Slow if n is very large |

With normal equation, computing $(X^TX)^{-1}$ is very slow. In practice, when n exceeds 10,000 it recommended to use gradient descent.

### Normal Equation Noninvertibility

When computing $\theta = (X^TX)^{-1}X^Ty$, what if $X^TX$ is non-invertible? (singular/degenerate)

Octave: `pinv(X'*X)*X'*y`

Usually, when the matrix is non-invertible, it is because of the following:

* Features are redundant.
  * Example $x\_1 = \text{size in feet}^2 \text{, } x\_2 = \text{size in m}^2$
  * Solve by deleting the linearly dependent features
* Too many features ($m \geq n$)
  * Solve by deleting some features or use regularization

# Optional Octave/MatLab Tutorial

## Octave Tutorial

### Basic Operations

```octave
% Basic Math operations
5 + 6
% ans =  11
3 - 2
% ans =  1
5 * 8
% ans =  40
1/2
% ans =  0.50000
2^6
% ans =  64

% Basic Logic operations
1 == 2   % does 1 equal 2, false
% ans = 0
1 ~= 2   % does 1 not equal 2, true
% ans = 1
1 && 0  % AND
% ans = 0
1 || 0  % OR
% ans = 1
xor(1, 0)
% ans = 1
```

If using the terminal:

```octave
octave:1>
octave:1> PS1('>> ');
>>

% Variables

>> a = 3
a =  3
>> b = 'hi'
b = hi
>> c = (3>=1)
c = 1
>> d = pi;
>> d
d =  3.1416
>> disp(d)
 3.1416
>> disp(sprintf('2 decimals: %0.2f', d))
2 decimals: 3.14
>> disp(sprintf('6 decimals: %0.6f', d))
6 decimals: 3.141593
>> d
d =  3.1416
>> format long
>> d
d =  3.14159265358979

% Matrices

>> A = [1 2; 3 4; 5 6;]
A =

   1   2
   3   4
   5   6

>> A = [1 2;
> 3 4;
> 5 6;]
A =

   1   2
   3   4
   5   6

>> v = [1 2 3]
v =

   1   2   3

>> v = [1; 2; 3;]
v =

   1
   2
   3

>> v = 1:0.1:2
v =

 Columns 1 through 4:

    1.00000000000000    1.10000000000000    1.20000000000000    1.30000000000000

 Columns 5 through 8:

    1.40000000000000    1.50000000000000    1.60000000000000    1.70000000000000

 Columns 9 through 11:

    1.80000000000000    1.90000000000000    2.00000000000000

>> v = 1:6
v =

   1   2   3   4   5   6

>> ones(2, 3)
ans =

   1   1   1
   1   1   1

>> C = 2*ones(2, 3)
C =

   2   2   2
   2   2   2

>> w = ones(1, 3)
w =

   1   1   1

>> w = zeros(1, 3)
w =

   0   0   0

>> w = rand(2, 4)
w =

   0.960581381809874   0.907561746558675   0.602424636369362   0.196393810472357
   0.544523059767745   0.133228586828452   0.775156655765407   0.684458260545384

>> w = randn(2, 4) % gaussian random numbers
w =

  -0.0963175155034235  -1.8261350128469240  -0.8791045598544875   1.6757509151763539
  -0.9677806051392197   0.9803503731745844   0.8030214909311352  -1.0019368744201753

>> w = -6 + sqrt(10)*(randn(1, 10000));  % semicolon suppresses output to console
>> hist(w) % see below
>> hist(w, 50) % see below
>> eye(4) % Identity function
ans =

Diagonal Matrix

   1   0   0   0
   0   1   0   0
   0   0   1   0
   0   0   0   1

>> help eye % documentation for eye
>> help % general octave help
```
![octave_basic_operations_1](/img/coursera-machine-learning-week2/octave_basic_operations_1.png)
Output of `hist(w)`, see how the random numbers generated follow gaussian distribution?

![octave_basic_operations_2](/img/coursera-machine-learning-week2/octave_basic_operations_2.png)
Output of `hist(w, 50)`, more buckets.

### Moving Data Around

```octave
>> A = [1 2; 3 4; 5 6;]
A =

   1   2
   3   4
   5   6

>> size(A)
ans =

   3   2

>> sz = size(A)
sz =

   3   2

>> size(sz)
ans =

   1   2

>> size(A, 1)
ans =  3
>> size(A, 2)
ans =  2
>> v
v =

   1   2   3   4   5   6

>> length(v)
ans =  6
>> length(A)
ans =  3

% Moving around the filesystem

>> pwd
ans = /Users/udia/sandbox/src/github.com/awwong1/alexander-wong.com
>> cd ~
>> pwd
ans = /Users/udia
>> ls
Applications				Music
Creative Cloud Files			Pictures
Desktop					Public
Documents				VirtualBox VMs
Downloads				anaconda
Library

% Loading data

>> load featuresX.dat
>> load('featuresX.dat')
```

```octave

octave:1> who
octave:2> A = [1; 2; 3]
A =

   1
   2
   3

octave:3> who
Variables in the current scope:

A

octave:4> whos
Variables in the current scope:

   Attr Name        Size                     Bytes  Class
   ==== ====        ====                     =====  =====
        A           3x1                         24  double

Total is 3 elements using 24 bytes

octave:5> clear A
octave:6> whos % displays nothing
octave:7> w = -6 + sqrt(10)*(randn(1, 10000));
octave:8> whos
Variables in the current scope:

   Attr Name        Size                     Bytes  Class
   ==== ====        ====                     =====  =====
        w           1x10000                  80000  double

Total is 10000 elements using 80000 bytes

octave:9> save dist.dat w;  % saves the data as binary format to dist.dat
octave:10> ls
LICENSE		config.toml	dist.dat	static
README.md	content		docs		themes
archetypes	data		layouts
octave:11> clear w
octave:12> whos
octave:13> load dist.dat w;
octave:14> whos
Variables in the current scope:

   Attr Name        Size                     Bytes  Class
   ==== ====        ====                     =====  =====
        w           1x10000                  80000  double

Total is 10000 elements using 80000 bytes

octave:15> save data.txt w -ascii  % saves the data as ascii format to dist.txt
octave:16> A = [1 2; 3 4; 5 6]
A =

   1   2
   3   4
   5   6

octave:17> A(3, 2)
ans =  6
octave:18> A(2, :) % ":" means every element along that row/column
ans =

   3   4
  
octave:19> A(:, 2)
ans =

   2
   4
   6

octave:20> A([1 3], :) % Get the entire row of the first and third rows
ans =

   1   2
   5   6

octave:21> A(:, 2) = [10; 11; 12] % assign 10, 11, 12 to the second column of A
A =

    1   10
    3   11
    5   12

octave:22> A = [A, [100; 101; 102;]] % Append the column 100, 101, 102 to the right of A
A =

     1    10   100
     3    11   101
     5    12   102

octave:23> A(:) % Put all elements of A into a single vector
ans =

     1
     3
     5
    10
    11
    12
   100
   101
   102

octave:24> A = [1 2; 3 4; 5 6;]
A =

   1   2
   3   4
   5   6

octave:25> B = [11 12; 13 14; 15 16]
B =

   11   12
   13   14
   15   16

octave:26> C = [A B]  % Concatenate matrices together (left-right)
C =

    1    2   11   12
    3    4   13   14
    5    6   15   16

octave:27> C = [A; B] % Concatenate matrices together (top-bottom)
C =

    1    2
    3    4
    5    6
   11   12
   13   14
   15   16
```

### Computing on Data

```octave

octave:1> A = [1 2; 3 4; 5 6];
octave:2> B = [11 12; 13 14; 15 16];
octave:3> C = [1 1; 2 2];
octave:4> A * C
ans =

    5    5
   11   11
   17   17

octave:5> A .* B % element-wise multiplication
ans =

   11   24
   39   56
   75   96

octave:6> A .^ 2 % element-wise squaring
ans =

    1    4
    9   16
   25   36

octave:7> v = [1; 2; 3]
v =

   1
   2
   3

octave:8> 1 ./v
ans =

   1.00000
   0.50000
   0.33333

octave:9> 1 ./ A
ans =

   1.00000   0.50000
   0.33333   0.25000
   0.20000   0.16667

octave:10> log(v)
ans =

   0.00000
   0.69315
   1.09861

octave:11> exp(v)
ans =

    2.7183
    7.3891
   20.0855

octave:12> abs([-1; -2; -3])
ans =

   1
   2
   3

octave:13> -v
ans =

  -1
  -2
  -3

octave:14> v + ones(length(v), 1)
ans =

   2
   3
   4

octave:15> v + 1
ans =

   2
   3
   4

octave:16> A
A =

   1   2
   3   4
   5   6

octave:17> A'
ans =

   1   3   5
   2   4   6

octave:18> a = [1 15 2 0.5]
a =

    1.00000   15.00000    2.00000    0.50000

octave:19> val = max(a)
val =  15
octave:20> [val, ind] = max(a)
val =  15
ind =  2
octave:21> a < 3
ans =

  1  0  1  1

octave:22> find (a < 3)
ans =

   1   3   4

octave:23> A = magic(3) % magic function returns magic squares. mathematical property that all rows, columns, diagonals sum up to the same thing
A =

   8   1   6
   3   5   7
   4   9   2

octave:24> [r, c] = find(A >= 7)
r =

   1
   3
   2

c =

   1
   2
   3

octave:25> a
a =

    1.00000   15.00000    2.00000    0.50000

octave:26> sum(a)
ans =  18.500
octave:27> prod(a)
ans =  15
octave:28> floor(a)
ans =

    1   15    2    0

octave:29> ceil(a)
ans =

    1   15    2    1

octave:30> rand(3)
ans =

   0.353560   0.209441   0.277491
   0.976008   0.349771   0.147595
   0.087618   0.057337   0.111061

octave:31> max(rand(3), rand(3)) % element wise max of two 3 by 3 matrices
ans =

   0.99776   0.67772   0.72626
   0.82404   0.99913   0.97748
   0.59569   0.99710   0.78396

octave:33> max(A, [], 1) % take the maximum per row
ans =

   8   9   7

octave:34> max(A, [], 2) % take the maximum per column
ans =

   8
   7
   9

octave:35> A = magic(5)
A =

   17   24    1    8   15
   23    5    7   14   16
    4    6   13   20   22
   10   12   19   21    3
   11   18   25    2    9

octave:36> sum(A, 1) % sum the columns
ans =

   65   65   65   65   65

octave:37> sum(A, 2) % sum the rows
ans =

   65
   65
   65
   65
   65

octave:38> sum(sum(A .* eye(5))) % sum the diagonals from top left to bottom right
ans =  65
octave:39> sum(sum(A .* flipud(eye(5)))) % sum the diagonals from the top right to bottom left
ans =  65
octave:40> A = magic(3)
A =

   8   1   6
   3   5   7
   4   9   2

octave:41> pinv(A)
ans =

   0.147222  -0.144444   0.063889
  -0.061111   0.022222   0.105556
  -0.019444   0.188889  -0.102778

octave:42> A * pinv(A)
ans =

   1.0000e+00  -1.2323e-14   6.6613e-15
  -6.9389e-17   1.0000e+00   2.2204e-16
  -5.8564e-15   1.2434e-14   1.0000e+00
```

### Plotting Data

```octave
octave:1> t=[0:0.01:0.98];
octave:2> y1 = sin(2*pi*4*t);
octave:3> plot(t, y1)
```

![octave_plotting_data_1](/img/coursera-machine-learning-week2/octave_plotting_data_1.png)
Horizontal axis is the `t` variable, the vertical axis is `y1`

```octave
octave:4> y2 = cos(2*pi*4*t);
octave:5> plot(t, y1);
octave:6> hold on;
octave:7> plot(t, y2, 'r');
octave:8> xlabel('time');
octave:9> ylabel('value');
octave:10> legend('sin', 'cos');
octave:11> title('my plot')
octave:12> print -dpng 'myPlot.png'
octave:13> close
```

![octave_my_plot](/img/coursera-machine-learning-week2/octave_my_plot.png)
Observe the labels, legends, title.

```octave
octave:14> figure(1); plot(t, y1);
octave:15> figure(2); plot(t, y2);
octave:16> subplot(1, 2, 1) % divides plot into a 1x2 grid, accessing the fist element
octave:17> plot(t, y1)
octave:18> subplot(1, 2, 2);
octave:19> plot(t, y2)
octave:20> close
```

![octave_plotting_data_2](/img/coursera-machine-learning-week2/octave_plotting_data_2.png)
Subplot allows showing two plots side by side

```octave
octave:21> axis([0.5 1 -1 1]) % change the x and y axis
octave:22> clf % clear the plot screen
octave:23> A = magic(5)
A =

   17   24    1    8   15
   23    5    7   14   16
    4    6   13   20   22
   10   12   19   21    3
   11   18   25    2    9

octave:24> imagesc(A)
octave:25> clf;
octave:26> imagesc(A)
octave:27> imagesc(A), colorbar, colormap gray;
```

![octave_plotting_data_3](/img/coursera-machine-learning-week2/octave_plotting_data_3.png)

### Functions & Control Statements: for, while, if/elseif/else

```octave
octave:1> PS1('>> ');
>> v = zeros(10, 1)
v =

   0
   0
   0
   0
   0
   0
   0
   0
   0
   0

% for loop
>> for i=1:10,
>   v(i) = 2^i;
>  end;
>> v
v =

      2
      4
      8
     16
     32
     64
    128
    256
    512
   1024

% while loop
>> i = 1;
>> while i <= 5,
>   v(i) = 100;
>   i = i + 1;
>  end;
>> v
v =

    100
    100
    100
    100
    100
     64
    128
    256
    512
   1024

% while loop with an if statement break
>> i = 1;
>> while true,
>   v(i) = 999;
>   i = i+1;
>   if i == 6,
>     break;
>   end;
>  end;
>> v
v =

    999
    999
    999
    999
    999
     64
    128
    256
    512
   1024

% If statement, with elseif and else
>> v(1) = 2;
>> if v(1) == 1,
>   disp('The value is one');
>  elseif v(1) == 2,
>   disp('The value is two');
>  else
>   disp('The value is not one or two.');
>  end;
The value is two
```

Functions in octave are defined by creating a file. For example, consider a file named `squareThisNumber.m` with the following:

```octave
% contents of squareThisNumber.m
function y = squareThisNumber(x)
% Return one variable 'y', has one input variable 'x'

y = x^2;
```

```octave
>> cd \...\ %where directory contains squareThisNumber.m
>> squareThisNumber(5)
ans = 25
```

Consider a file named `squareAndCubeThisNumber.m` with the following:

```octave
% contents of squareAndCubeThisNumber.m
function [y1, y2] = squareAndCubeThisNumber(x)
% Return two variables 'y1', 'y2'; has one input variable 'x'

y1 = x^2;
y2 = x^3;
```

```octave
>> cd \...\ %where directory contains squareAndCubeThisNumber.m
>> [a, b] = squareAndCubeThisNumber(5)
>> a
a = 25
>> b
b = 125
```

Example:

```octave
% costFunctionJ.m
function J = costFunctionJ(X, y, theta)

% X is the "design matrix" containing the training examples.
% y is the class labels

m = size(X, 1);          % number of training examples
predictions = X * theta; % predictions of hypothesis on all m examples
sqrErrors = (predictions - y) .^ 2 % squared errors

J = 1/(2*m) * sum(sqrErrors);
```

```octave
octave:1> X = [1 1; 1 2; 1 3]
X =

   1   1
   1   2
   1   3

octave:2> y = [1; 2; 3]
y =

   1
   2
   3

octave:3> theta = [0;1]
theta =

   0
   1

octave:4> j = costFunctionJ(X, y, theta)
sqrErrors =

   0
   0
   0

j = 0

octave:5> j = costFunctionJ(X, y, [0; 0])
sqrErrors =

   1
   4
   9

j =  2.3333
```

### Vectorization

Example 1:

Consider the following hypothesis function

$$ h\_\theta(x) = \sum\limits\_{j=0}^n\theta\_jx\_j = \theta^Tx $$

```octave
% Unvectorized implementation
prediction = 0.0;
for j = 1:n+1,
  prediction = prediction + theta(j) * x(j)
end;

% Vectorized implementation
prediction = theta' * x;
```

Example 2:

Recall the gradient descent algorithm

$$ \theta\_j := \theta\_j - \alpha \dfrac{1}{m} \sum\limits\_{i=1}^m(h\_\theta(x^{(i)}) - y^{(i)})\cdot x\_j^{(i)} \hspace{2em} \text{(for all } j \text{)}$$

_Unvectorized naive approach_

$$ \theta\_0 := \theta\_0 - \alpha \dfrac{1}{m} \sum\limits\_{i=1}^m(h\_\theta(x^{(i)}) - y^{(i)})\cdot x\_0^{(i)} $$
$$ \theta\_1 := \theta\_1 - \alpha \dfrac{1}{m} \sum\limits\_{i=1}^m(h\_\theta(x^{(i)}) - y^{(i)})\cdot x\_1^{(i)} $$
$$ \theta\_2 := \theta\_2 - \alpha \dfrac{1}{m} \sum\limits\_{i=1}^m(h\_\theta(x^{(i)}) - y^{(i)})\cdot x\_2^{(i)} $$
$$ \vdots $$
$$ \theta\_n := \theta\_n - \alpha \dfrac{1}{m} \sum\limits\_{i=1}^m(h\_\theta(x^{(i)}) - y^{(i)})\cdot x\_n^{(i)} $$

_Vectorized approach_

$$ \text{Simply} \hspace{2em} \Theta := \Theta - \alpha\delta $$
$$ \Theta \leftarrow \mathbb{R}^{n+1} $$
$$ \alpha \leftarrow \mathbb{R} $$
$$ \delta \leftarrow \mathbb{R}^{n+1} $$
$$ \text{where} \hspace{2em} \delta = \dfrac{1}{m} \sum\limits\_{i=1}^m (h\_\theta(x^{(i)})-y^{(i)}) \cdot x^{(i)} $$
$$ (h\_\theta(x^{(i)}) - y^{(i)}) \leftarrow \mathbb{R} $$
$$ x^{(i)} \leftarrow \mathbb{R}^{n+1} $$

Recall that $ \mathbb{R} $ means a scalar number and $\mathbb{R}^{n+1}$ is a vector of scalar numbers.

---

Week 3 tbd.
