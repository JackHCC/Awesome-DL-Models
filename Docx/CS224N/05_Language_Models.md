# Recurrent Neural Networks and Language Models

## Simple feed-forward neural network multi-class classifier

![](../../Images/CS224N/image-20220125185406373.png)

## A bit more about neural networks

- Regularization
- Dropout
- Vectorization（Matrices are awesome）
- Non-linearities, activate function
- Parameter Initialization
- Optimizers
- Learning Rates

## Language Modeling + RNNs

### Language Modeling

**Language Modeling** is the task of predicting what word comes next

![](../../Images/CS224N/image-20220125190502545.png)

![](../../Images/CS224N/image-20220125190526891.png)

### n-gram Language Models

Definition: A n-gram is a chunk of n consecutive words.

- unigrams: “the”, “students”, “opened”, ”their” 
- bigrams: “the students”, “students opened”, “opened their” 
- trigrams: “the students opened”, “students opened their” 
- 4-grams: “the students opened their”

![](../../Images/CS224N/image-20220125191042360.png)

**Sparsity Problems with n-gram Language Models**

![](../../Images/CS224N/image-20220125191131528.png)

**A fixed-window neural Language Model**

![](../../Images/CS224N/image-20220125191310820.png)

### Recurrent Neural Networks (RNN)

![](../../Images/CS224N/image-20220125191415683.png)



## Links

- [Note](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (blog post overview)
- [On Chomsky and the Two Cultures of Statistical Learning](http://norvig.com/chomsky.html)













































































