# Transformers

## From recurrence (RNN) to attention-based NLP models

### Issues with recurrent models

**Linear interaction distance**

- O(sequence length) steps for distant word pairs to interact means: 
  - Hard to learn long-distance dependencies (because gradient problems!) 
  - Linear order of words is “baked in”; we already know linear order isn’t the  right way to think about sentences…

**Lack of parallelizability**

- Forward and backward passes have O(sequence length)  unparallelizable operations
  - GPUs can perform a bunch of independent computations at once! 
  - But future RNN hidden states can’t be computed in full before past RNN  hidden states have been computed
  - Inhibits training on very large datasets!

**How about word windows?**

- Word window models aggregate local contexts 
  - (Also known as 1D convolution; we’ll go over this in depth later!) 
  - Number of unparallelizable operations does not increase sequence length!

- Word window models aggregate local contexts
- What about long-distance dependencies?
  - Stacking word window layers allows interaction between farther words
- Maximum Interaction distance = sequence length / window size
  - (But if your sequences are too long, you’ll just ignore long-distance context)x

![](../../Images/CS224N/image-20220129162304159.png)

**How about attention?**

- Attention treats each word’s representation as a query to access and  incorporate information from a set of values. 
  - We saw attention from the decoder to the encoder; today we’ll think about  attention within a single sentence.
- Number of unparallelizable operations does not increase sequence length. 
- Maximum interaction distance: O(1), since all words interact at every layer!

![](../../Images/CS224N/image-20220129162355961.png)

### Self-Attention

![](../../Images/CS224N/image-20220129162422853.png)

#### sequence order

- **Sinusoidal position representations**: concatenate sinusoidal functions of varying periods:

![](../../Images/CS224N/image-20220129162525029.png)

#### Adding nonlinearities

![](../../Images/CS224N/image-20220129162904696.png)



#### Masking the future

![](../../Images/CS224N/image-20220129162942764.png)



### Barriers and solutions for Self-Attention as a building block

![](../../Images/CS224N/image-20220129163004561.png)

### Necessities for a self-attention building block

- Self-attention: 
  - the basis of the method. 
- Position representations:
  - Specify the sequence order, since self-attention is an unordered function of its  inputs. 
- Nonlinearities: 
  - At the output of the self-attention block
  - Frequently implemented as a simple feed-forward network. 
- Masking: 
  - In order to parallelize operations while not looking at the future. 
  - Keeps information about the future from “leaking” to the past.

## Introducing the Transformer model 

### The Transformer Encoder-Decoder [Vaswani et al., 2017]

![](../../Images/CS224N/image-20220129163138979.png)

What’s left in a Transformer Encoder Block that we haven’t covered? 

1. **Key-query-value attention**: How do we get the 𝑘, 𝑞, 𝑣 vectors from a single word embedding?
2. **Multi-headed attention**: Attend to multiple places in a single layer! 
3. **Tricks to help with training**! 
   1. Residual connections 
   2. Layer normalization 
   3. Scaling the dot product 
   4. These tricks don’t improve what the model is able to do; they help improve the training process.  Both of these types of modeling improvements are very important!

### Key-Query-Value Attention

![](../../Images/CS224N/image-20220129163303484.png)

![](../../Images/CS224N/image-20220129163315623.png)

### Multi-headed attention

![](../../Images/CS224N/image-20220129163336713.png)

![](../../Images/CS224N/image-20220129163357657.png)

### Residual connections

![](../../Images/CS224N/image-20220129163419780.png)

### Layer normalization

![](../../Images/CS224N/image-20220129163559440.png)

### Scaled Dot Product

![](../../Images/CS224N/image-20220129163619372.png)

### The Transformer Encoder-Decoder

![](../../Images/CS224N/image-20220129163737704.png)

![](../../Images/CS224N/image-20220129163705800.png)

### Cross-attention

![](../../Images/CS224N/image-20220129163802913.png)

## Drawbacks and variants of Transformers

### Quadratic computation as a function of sequence length

![](../../Images/CS224N/image-20220129163925524.png)

### Recent work on improving on quadratic self-attention cost

![](../../Images/CS224N/image-20220129163954117.png)

![](../../Images/CS224N/image-20220129164007100.png)

## Links

- [Note](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes07-QA.pdf)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformer (Google AI blog post)](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)



