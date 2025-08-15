# Theory behind our neural network


## ReLU layer 

ReLU stands for **Rectified Linear Unit**, and it's one of the most common activation functions in deep learning.

---

### 1. Definition

Mathematically:

$$
\text{ReLU}(x) = \max(0, x)
$$

* If the input $x > 0$, output is $x$.
* If the input $x \le 0$, output is $0$.

Example:

| Input $x$ | Output ReLU(x) |
| --------- | -------------- |
| -2        | 0              |
| 0         | 0              |
| 1.5       | 1.5            |
| 10        | 10             |

---

### 2. Purpose

In a neural network, activation functions introduce **non-linearity**, allowing the network to learn complex relationships between inputs and outputs.
ReLU is popular because:

* **Simple computation** → just a max operation.
* **Avoids vanishing gradients** (better than sigmoid/tanh for deep networks).
* **Sparse activation** → only positive inputs pass through, which can help generalization.

---

### 3. Graph

```
y
↑
│        /
│       /
│      /
│     /
└─────┬─────────→ x
      0
```

The function is **flat** for $x \le 0$ and a straight line with slope 1 for $x > 0$.

---

### 4. Pros and Cons

**Pros**

* Efficient to compute.
* Helps deep networks train faster.
* Reduces the risk of vanishing gradients.

**Cons**

* Can cause the **"dying ReLU" problem**: neurons stuck with outputs at zero stop learning if weights make all inputs negative.
* Unbounded positive side can sometimes lead to exploding activations (handled with normalization).

---

### 5. Variants

To fix ReLU's drawbacks, researchers have developed:

* **Leaky ReLU** → small slope for $x < 0$ instead of 0.
* **Parametric ReLU (PReLU)** → slope learned during training.
* **ELU** / **GELU** → smoother curves for better gradient flow.

---

## why use only relu after the first fully connected layer here

In that architecture, the **ReLU after the first fully connected (FC) layer** serves a very specific role:

---

### 1. **Breaking linearity**

* If you had **two fully connected layers back-to-back with no non-linearity**, they'd mathematically collapse into a *single* linear transformation.
* This would mean your network—no matter how many such layers—would act just like **one big linear model**.
* By placing a **ReLU** between `fc` (128 neurons) and `fc_1` (10 neurons), you insert non-linearity so the model can learn complex, non-linear relationships in the data.

---

### 2. **Why not after the second FC layer?**

* The second FC layer (`fc_1`) is followed immediately by **Softmax**, which already applies a non-linear transformation to produce class probabilities.
* Adding ReLU before softmax could:

  * Zero out negative class scores unnecessarily.
  * Distort the intended raw logits that softmax expects.
* Therefore, the **last hidden layer** is where you usually put your final activation (ReLU here), and the **output layer** uses softmax (or another output-specific activation).

---

### 3. **Flow of transformations here**

1. **ImageInput (28×28)** → flattened into a 784-D vector.
2. **FC(784→128)** → dense representation of input features.
3. **ReLU** → keeps positive activations, kills negatives, introduces non-linearity.
4. **FC(128→10)** → computes class scores for each category.
5. **Softmax** → converts scores into probabilities.

---

### 4. **Key takeaway**

The ReLU here:

* Makes sure the network isn't just a big linear function.
* Lets the first FC layer learn richer, more abstract features before classification.
* Is skipped after the final FC layer because the classification step uses softmax instead.

---

If you want, I can sketch a **MATLAB-style data flow diagram** of this network showing exactly where and why ReLU fits in. That could make the reasoning even clearer.
