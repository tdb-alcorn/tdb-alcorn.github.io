---
layout: post
date:   2017-12-17 15:59:53 -0800
header:
    image: /assets/img/small-wreck-tom.jpg
---
Perceptrons were one of the earliest supervised learning models, and though they went unloved for about 50 years, they've recently experienced a huge comeback as "multi-layer perceptrons" a.k.a. neural networks. In this post I want to explore how some visualisations of what a perceptron "sees", which will help us understand why chaining together multiple layers of perceptrons is such an effective trick.

Briefly, a perceptron is a type of linear model that learns a decision boundary by averaging together data points until it reaches a stable solution. For the purposes of visualisation I'll only be considering perceptrons with 2 inputs, but the intuition we'll develop applies equally well to higher dimensional data (although it is substantially trickier to visualise). So, for our purposes, a perceptron is a function of an input vector that produces a prediction (also called an activation). You will usually see this function written as

$$\hat{y} = \sigma(Wx + b)$$

where $$\hat{y}$$ is the predicted value, $$\sigma$$ is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function), $$W$$ is a vector of weights, $$x$$ is the input vector and $$b$$ is a bias term. In the 2 input case this breaks down as

$$\hat{y} = \sigma(w_0 x_0 + w_1 x_1 + b)$$

Essentially, a perceptron takes a weighted average of its inputs and then computes the sigmoid of the result, yielding a result between 0 and 1. The weighted average part seems reasonable enough, but why compute a sigmoid? This may seem myterious and arbitrary to you right now, but we are going pry open the black box to see why the sigmoid activation function works very well.

For a two input perceptron, the decision boundary is a contour (line of constant value) of a function of the two inputs. 



```python
import matplotlib
%matplotlib inline
matplotlib.rcParams['figure.dpi']= 300
import matplotlib.pyplot as plt
import numpy as np
```


```python
np.random.seed(40)
data = np.random.normal(size=40).reshape((-1, 2))

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def perceptron(weights, x):
    return sigmoid(np.dot(weights, x))

def decision_boundary(weights, x0):
    return -1. * weights[0]/weights[1] * x0
```

I define two perceptrons that divide the data in two different ways. We will see how they combine to transform the data into a new space in which they can be linearly separated. The bias won't qualitatively change the following visualisations so I am going to set it to 0 to keep things simple.


```python
label = np.apply_along_axis(lambda x: 1 if x[0] > 0 and x[1] > 0 else 0, 1, data)
w0 = np.array([10, -1])
w1 = np.array([-1, 10])

data_0 = data[label == 0]
data_1 = data[label == 1]
t0 = np.linspace(-0.2, 0.2, num=50)
t1 = np.linspace(-2.5, 2.5, num=50)

fig = plt.figure(figsize=(15,7))

ax = plt.subplot(1,2,1)
ax.scatter(data_0[:, 0], data_0[:, 1], c='b', label='Class #0')
ax.scatter(data_1[:, 0], data_1[:, 1], c='r', label='Class #1')
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")
ax.legend()

ax = plt.subplot(1,2,2)
ax.scatter(data_0[:, 0], data_0[:, 1], c='b', label='Class #0')
ax.scatter(data_1[:, 0], data_1[:, 1], c='r', label='Class #1')
ax.plot(t0, decision_boundary(w0, t0), 'm', label='Perceptron #0 decision boundary')
ax.plot(t1, decision_boundary(w1, t1), 'g', label='Perceptron #1 decision boundary')
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")
ax.legend()
```



![png]({{ "/assets/2017-12-17-seeing-like-a-perceptron/output_4_1.png" | absolute_url }})


Perceptron \#0 classifies everything above the line as 1 and everything below the line as 0, and perceptron \#1 classifies everything to the right as 1 and everything to the left as 0. You can see that on their own, neither perceptron does a very good job of classifying the data, but there is a region, the upper right, in which they both get most of their predictions correct. If we could somehow combine these decision boundaries we might have a reasonably good classifier...

And this is precisely what a multi-layer perceptron does! We'll make a second layer that combines the outputs of the first layer (the outputs of perceptrons \#0 and \#1) to make a better prediction than either classifier in the first layer could have independently come up with.

But as we know, a perceptron is a linear model: the only thing it can do is draw a line through the input space that hopefully separates the data into two classes. This means that if our second layer makes good predictions, the inputs that it received must have been linearly separable (or nearly so). Therefore, we can deduce that the first layer must have performed some kind of transformation on the original data that yielded a new set of data that was linearly separable! A natural question, then, is what exactly do the inputs to the second layer look like? Or, to put it more precisely, what does our input space look like after the first layer has applied its transformation?


```python
nl_transform = lambda d: np.apply_along_axis(lambda x: [perceptron(w0, x), perceptron(w1, x)], 1, d)

data_t = nl_transform(data)
data_t_0 = data_t[label == 0]
data_t_1 = data_t[label == 1]

db0 = nl_transform(np.array([t0, decision_boundary(w0, t0)]).T)
db1 = nl_transform(np.array([t1, decision_boundary(w1, t1)]).T)

fig = plt.figure(figsize=(10,10))
ax = plt.axes()
ax.scatter(data_t_0[:, 0], data_t_0[:, 1], c='b', label='Class #0')
ax.scatter(data_t_1[:, 0], data_t_1[:, 1], c='r', label='Class #1')
ax.plot(db0[:, 0], db0[:, 1], 'm', label='Perceptron #0 decision boundary')
ax.plot(db1[:, 0], db1[:, 1], 'g', label='Perceptron #1 decision boundary')
ax.set_xlabel('Perceptron #0 output')
ax.set_ylabel('Perceptron #1 output')
ax.legend()
```



![png]({{ "/assets/2017-12-17-seeing-like-a-perceptron/output_7_1.png" | absolute_url }})


Holy smokes! This looks promising, because we can immediately see that the data has been squished out to the sides of the plot and is now linearly separable. Additionally, something interesting has happened to the decision boundaries: they have become like a set of axes (an orthogonal basis for the space). Could we have guessed that this would happen? It turns out we could have: remember that a decision boundary is just a line along which the perceptron predicts a constant value, called the threshold. For example, if our decision threshold is 0.5 then the decision boundary is the set of points along which the perceptron outputs $$\hat{y}$$ = 0.5. Now, since the predictions of each perceptron make up the coordinates of our new space, we see the decision boundaries as lines of constant value in each coordinate positioned at the decision threshold.

To get some more intuition for what this transformation has actually done, it would be nice to see where the gridlines from the original input space ended up in this new space. That is to say, what happened to our original axes?


```python
n_gridlines = 100
gridline_x = np.linspace(-2.5, 2.5, num=n_gridlines)
gridline_y = np.linspace(-1.5, 1.5, num=n_gridlines)

fig = plt.figure(figsize=(15,7))

ax = plt.subplot(1,2,1)
for glx in gridline_x:
    g_t = np.array([np.zeros(n_gridlines) + glx, gridline_y]).T
    ax.plot(g_t[:, 0], g_t[:, 1], color='xkcd:rust', linewidth=0.5, zorder=1)
for gly in gridline_y:
    g_t = np.array([gridline_x, np.zeros(n_gridlines) + gly]).T
    ax.plot(g_t[:, 0], g_t[:, 1], color='xkcd:deep blue', linewidth=0.5, zorder=1)
ax.scatter(data_0[:, 0], data_0[:, 1], c='b', label='Class #0', zorder=2)
ax.scatter(data_1[:, 0], data_1[:, 1], c='r', label='Class #1', zorder=2)
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")

ax = plt.subplot(1,2,2)
for glx in gridline_x:
    g_t = nl_transform(np.array([np.zeros(n_gridlines) + glx, gridline_y]).T)
    ax.plot(g_t[:, 0], g_t[:, 1], color='xkcd:rust', linewidth=0.5, zorder=1)
for gly in gridline_y:
    g_t = nl_transform(np.array([gridline_x, np.zeros(n_gridlines) + gly]).T)
    ax.plot(g_t[:, 0], g_t[:, 1], color='xkcd:deep blue', linewidth=0.5, zorder=1)
ax.scatter(data_t_0[:, 0], data_t_0[:, 1], c='b', label='Class #0', zorder=2)
ax.scatter(data_t_1[:, 0], data_t_1[:, 1], c='r', label='Class #1', zorder=2)
ax.set_xlabel('Perceptron #0 output')
ax.set_ylabel('Perceptron #1 output')
```



![png]({{ "/assets/2017-12-17-seeing-like-a-perceptron/output_9_1.png" | absolute_url }})


Woah! Here the $$y$$ (vertical) gridlines from the input space are plotted in red and the $$x$$ (horizontal) gridlines are plotted in blue. So what does this tell us? First, notice that the entirety of our two-dimensional space (all of $$\mathbb{R}^2$$) has been mapped into a 1x1 square. This is a consequence of the sigmoid function, which only outputs values between 0 and 1. Most of the new space is filled up with gridlines that were close to the decision boundaries in the input space. All the other gridlines (extending off to $$\pm \infty$$ in both directions) have been squished up against the sides of the 1x1 square, which explains why all the data is also concentrated around the edges. We can think of this as a kind of fish-eye effect: the new space magnifies the parts of the old space which were adjacent to the decision boundaries (like the middle of a fish-eye lens) and pushes everything else out of the way. Why does this work? Loosely, you can think of the space around the decision boundaries as the most "interesting" parts of the data: it's the space that the two perceptrons in the first layer are least confident about classifying. So it makes sense to enhance our view of that part of the space. Since a perceptron is more confident in its predictions the further you get away from a decision boundary, the space far away from a decision boundary is boring and predictable (pun intended), so we can safely ignore it.

Notice also that the gridlines are all S-shaped, like the sigmoid curve (which owes its name to its peculiar S-shape). In fact, this is a consequence of using the sigmoid as the activation function. So does the useful "fish-eye" transformation depend on our choice of activation function? Let's consider what would happen if we instead used a step function as the activation. Since the step function outputs only 0 or 1 for any input, every point in the input space would be mapped to one of four points in the new space: (0,0), (1,0), (0,1) and (1,1), the corners of the unit square. These correspond to the four possible output combinations of the two perceptrons in the first layer. For each input data point, the second layer perceptron will receive one of those four corners as input and can use only that information to decide how to classify the point. This reduces the problem to a statistical average problem: what fraction of points in each corner belong to each class? We can conclude that if we were to use step activation, some nuances in the data would certainly be lost, and sigmoid activation seems better. On the other hand, it is definitely possible for a two layer perceptron to achieve good accuracy on this particular data set using step activation.

Ok, where are we now? Our clever layer-1 perceptrons found a non-linear transformation of the input space that yields a new space in which the data is more linearly separable than before. In this new space, we can easily eyeball a linear solution to the above classification, and a perceptron should have no trouble finding it either. Here's one possible solution:


```python
def decision_boundary_bias(weights, bias, x0):
    return -1. * (weights[0]/weights[1] * x0 + bias/weights[1])

fig = plt.figure(figsize=(10,10))
ax = plt.axes()

t2 = np.linspace(0, 1, num=50)
w2 = np.array([1,1])
b2 = -1.2

ax.scatter(data_t_0[:, 0], data_t_0[:, 1], c='b', label='Class #0')
ax.scatter(data_t_1[:, 0], data_t_1[:, 1], c='r', label='Class #1')
ax.plot(t2, decision_boundary_bias(w2, b2, t2), color='xkcd:chocolate', label='Perceptron #2 decision boundary')
ax.set_xlabel('Perceptron #0 output')
ax.set_ylabel('Perceptron #1 output')
ax.legend()
```


![png]({{ "/assets/2017-12-17-seeing-like-a-perceptron/output_11_1.png" | absolute_url }})


We can visually verify that this final decision boundary will correctly classify all the points in the transformed space. What will this line have to look like in the input space? Scroll back up and take a look at how the data is laid out: the final decision boundary definitely can't be a straight line! In fact it's going to have to be very curved, pulling a tight right-angle turn near the point (0,0). We can visualise the final decision boundary by inverting the non-linear transformation that we applied to the data:


```python
def logit(x):
    # Logit is the inverse of the sigmoid
    return np.log(x / (1. - x))

def perceptron_bias(weights, bias, x):
    return sigmoid(np.dot(weights, x) + bias)

wt = np.array([w0, w1])
wt_inv = np.linalg.inv(wt)
nl_transform_inv = lambda d: np.apply_along_axis(lambda o: np.matmul(wt_inv, logit(o)), 1, d)

t2 = np.linspace(0, 1, num=100000)
db2 = nl_transform_inv(np.array([t2, decision_boundary_bias(w2, b2, t2)]).T)

fig = plt.figure(figsize=(10,6))
ax = plt.axes()
ax.scatter(data_0[:, 0], data_0[:, 1], c='b', label='Class #0')
ax.scatter(data_1[:, 0], data_1[:, 1], c='r', label='Class #1')
ax.plot(t0, decision_boundary(w0, t0), 'm', label='Perceptron #0 decision boundary')
ax.plot(t1, decision_boundary(w1, t1), 'g', label='Perceptron #1 decision boundary')
ax.plot(db2[:, 0], db2[:, 1], color='xkcd:chocolate', label='Perceptron #2 decision boundary')
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")
ax.legend()
```

![png]({{ "/assets/2017-12-17-seeing-like-a-perceptron/output_13_2.png" | absolute_url }})


I've also plotted the original decision boundaries so that you can compare them with the final decision boundary. From this vantage point, it's clear that the final decision boundary is something like a weighted blend of the two boundaries from the first layer. This "blending" idea is usually the way people talk about multi-layer perceptrons, but to me it's a little unsatisfying. It is hard to intuitively see how such a simple algorithm could be smart enough to figure out a way to shape and position this tricky curved decision boundary that separates the data in the input space. But if we look at what the perceptron in the second layer is actually "seeing", it's clear that it only has to solve the same problem that perceptrons always solve: find a line in space that separates the data. The space has changed, not the algorithm.

As an aside, if you take the blending analogy further then another way to think about a multi-layer perceptron is as a kind of [ensemble method](https://en.wikipedia.org/wiki/Ensemble_learning). An ensemble method is a technique for taking predictions from a bunch of lousy models and producing a kind of weighted average that is more correct more often than any of the individual models. One example is something called a random forest, where you create a group of decision trees, train each one on only a small subset of the data, and then use the trees' collective predictions as a "vote" to decide how to classify each point. Another example is a technique called [boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning) (the idea behind [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)), in which you train successive models on the data where the previous model made the most mistakes. In our case, we can think of the first layer of the perceptron as an ensemble of lousy perceptrons, except instead of just voting or averaging classifiers, we also do something a little like boosting: we focus in on the areas where the ensemble is least confident and train new models on those regions.

Alright, one last visualisation to complete the picture. The plot above shows the decision boundary of the final perceptron, which is really just a contour line along which it predicts a constant $$\hat{y}$$ = 0.5. However, when building a predictive model you have the flexibility of choosing any activation threshold you want for the decision boundary. So what do the other contour lines look like in the original input space?


```python
def perceptron_contour(weights, bias, x, y):
    return sigmoid(weights[0]*x + weights[1]*y + bias)

fig = plt.figure(figsize=(10,6))
ax = plt.axes()

x = np.linspace(-2.5, 2.5, num=200)
y = np.linspace(-1.5, 1.5, num=200)
X, Y = np.meshgrid(x, y)
Z = perceptron_contour(w2, b2, perceptron_contour(w0, 0, X, Y), perceptron_contour(w1, 0, X, Y))
contours = np.linspace(0.1,0.9, num=21)
cs = ax.contour(X, Y, Z, contours, linewidths=0.5)
plt.clabel(cs, inline=1, fontsize=10, fmt='%1.2f')

ax.scatter(data_0[:, 0], data_0[:, 1], c='b', label='Class #0')
ax.scatter(data_1[:, 0], data_1[:, 1], c='r', label='Class #1')

ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")

ax.legend()
```


![png]({{ "/assets/2017-12-17-seeing-like-a-perceptron/output_15_1.png" | absolute_url }})


Here you can see the $$\hat{y}$$ = 0.5 decision boundary as before, along with a range of other possible choices. Note that in this scenario, everything above and to the right of a given boundary would be classified as class \#1, and everything else as class \#0. In the transformed space, these lines correspond to straight lines parallel to the decision boundary of the final perceptron.

To summarise what we have found: the first layer of the perceptron actually defines a non-linear transformation that yields a new data-set that is more linearly separable by blowing up the area around the decision boundaries in the input space and squishing down the rest. And since we have an algorithm for training perceptrons, they can find this transformation _by themselves_. Then we can apply another perceptron to find the linear separation in the new, transformed space. I think that's super cool, and it gets at the heart of what machine learning really is. It's not some magical black box hack, nor is it a rough simulation of brain activity. It's a principled system for finding non-linear transformations that make the data easier to work with.
