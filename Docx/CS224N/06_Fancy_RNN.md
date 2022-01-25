# Vanishing Gradients, Fancy RNNs, Seq2Seq

## The Simple RNN Language Model

![](../../Images/CS224N/image-20220125192708059.png)

### Training an RNN Language Mode

![](../../Images/CS224N/image-20220125192733281.png)

## Evaluating Language Models

- perplexity

![](../../Images/CS224N/image-20220125192812732.png)

## Vanishing gradient intuition

![](../../Images/CS224N/image-20220125193004385.png)

## Gradient clipping: solution for exploding gradient

![](../../Images/CS224N/image-20220125193032378.png)

## Long Short-Term Memory RNNs (LSTMs)

- Hadamard product：矩阵对应位置相乘

![](../../Images/CS224N/image-20220125193102820.png)

![](../../Images/CS224N/image-20220125193143002.png)

## Bidirectional and Multi-layer RNNs: motivation

![](../../Images/CS224N/image-20220125193230090.png)

- Note: bidirectional RNNs are only applicable if you have access to the entire input  sequence 
  - They are not applicable to Language Modeling, because in LM you only have left  context available. 
- If you do have entire input sequence (e.g., any kind of encoding), bidirectionality is  powerful (you should use it by default)

### Multi-layer RNNs

![](../../Images/CS224N/image-20220125193341633.png)



## Links

- [Note](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf)
- [Sequence Modeling: Recurrent and Recursive Neural Nets](http://www.deeplearningbook.org/contents/rnn.html) (Sections 10.3, 10.5, 10.7-10.12)
- [Vanishing Gradients Jupyter Notebook](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/lectures/vanishing_grad_example.html) (demo for feedforward networks)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (blog post overview)







