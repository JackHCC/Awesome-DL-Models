# CS224 Natural Language Processing

Course notes and code: including formula derivation, course knowledge summary, homework and model implementation code.

Lesson Link：[CS224N Schedule](https://web.stanford.edu/class/cs224n/index.html#schedule) 👈 You Can get Slides from here.

## Lecture and Papers
- [Word Vectors](01_Word_Vectors.md)
  * [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (original word2vec paper)
  * [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (negative sampling paper)
- [Word Vectors 2 and Word Window Classification](02_Word_Vectors_2.md)
  - [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf) (original GloVe paper)
    - [Official Code](https://nlp.stanford.edu/projects/glove/)
  - [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016)
  - [Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036)
  - [A Latent Variable Model Approach to PMI-based Word Embeddings](http://aclweb.org/anthology/Q16-1028)
  - [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320)
  - [On the Dimensionality of Word Embedding](https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf)
  - **Appendix**: [VSM](https://blog.csdn.net/weixin_42398658/article/details/85063004)，[LSA](https://zhuanlan.zhihu.com/p/144367432)，[PMI](https://blog.csdn.net/cj151525/article/details/112804799#:~:text=PMI%20%EF%BC%88%20Pointwise%20Mutual%20Information%20%EF%BC%89%20%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9B%B8%E5%85%B3%E6%96%87%E7%8C%AE%E9%87%8C%E9%9D%A2%EF%BC%8C%E7%BB%8F%E5%B8%B8%E4%BC%9A%E7%94%A8%E5%88%B0%20PMI,%28x%2Cy%29p%20%28x%29p%20%28y%29%3Dlogp%20%28x%7Cy%29p%20%28x%29%3Dlogp%20%28y%7Cx%29p%20%28y%29%20%E5%9C%A8%E6%A6%82%E7%8E%87%E8%AE%BA%E4%B8%AD%EF%BC%8C%E6%88%91%E4%BB%AC%E7%9F%A5%E9%81%93%EF%BC%8C%E5%A6%82)，[N-garm](https://zhuanlan.zhihu.com/p/32829048)，[NNLM](https://blog.csdn.net/lilong117194/article/details/82018008)，[RNNLM](https://blog.csdn.net/rongsenmeng2835/article/details/108656674)，[SVD](https://zhuanlan.zhihu.com/p/29846048)
  
- [Backprop and Neural Networks](03_Backprop.md)
  - [Learning Representations by Backpropagating Errors](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf) (seminal Rumelhart et al. backpropagation paper)
  - [Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

- [Dependency Parsing](04_Dependency_Parsing.md)
  - [Incrementality in Deterministic Dependency Parsing](https://www.aclweb.org/anthology/W/W04/W04-0308.pdf)
  - [A Fast and Accurate Dependency Parser using Neural Networks](https://www.emnlp2014.org/papers/pdf/EMNLP2014082.pdf)
    - [Unofficial Code](https://github.com/akjindal53244/dependency_parsing_tf)
  - [Dependency Parsing](http://www.morganclaypool.com/doi/abs/10.2200/S00169ED1V01Y200901HLT002) (a book need to buy)
  - [Globally Normalized Transition-Based Neural Networks](https://arxiv.org/pdf/1603.06042.pdf)
  - [Universal Stanford Dependencies: A cross-linguistic typology](http://nlp.stanford.edu/~manning/papers/USD_LREC14_UD_revision.pdf)
  - **UD Standard**: [Universal Dependencies](http://universaldependencies.org/)
  - **Appendix**: [t-SNE](https://zhuanlan.zhihu.com/p/148170862)，[Understand Stanford Universal Dependencies](https://www.cnblogs.com/brooksj/articles/14191888.html)，[Beam Search](https://zhuanlan.zhihu.com/p/82829880)，[All NLP Task Evaluation](https://my.oschina.net/u/4316091/blog/4270383)

- [Recurrent Neural Networks and Language Models](05_Language_Models.md)
  - [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) (textbook chapter)
  - [Sequence Modeling: Recurrent and Recursive Neural Nets](http://www.deeplearningbook.org/contents/rnn.html) (Sections 10.1 and 10.2)
  
- [Vanishing Gradients, Fancy RNNs, Seq2Seq](06_Fancy_RNN.md)
  - [Learning long-term dependencies with gradient descent is difficult](https://readpaper.com/pdf-annotate/note?noteId=645919043856969728&pdfId=4508884458182172673) (one of the original vanishing gradient papers)
  - [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/pdf/1211.5063.pdf) (proof of vanishing gradient problem)
  - [**Vanishing Gradient Solutions**](https://analyticsindiamag.com/addressing-the-vanishing-gradient-problem-a-guide-for-beginners/): Multi-Level Hierarchy, Long Short-Term Memory(LSTM)/Residual Neural Network, Rectified Linear Unit (ReLU) Activation Function
  
- [Machine Translation, Attention, Subword Models](07_Machine_Translation.md)
  - [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf) (original paper)
  - [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf) (original seq2seq NMT paper)
  - [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/pdf/1211.3711.pdf) (early seq2seq speech recognition paper)
  - [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) (original seq2seq+attention paper)
  - [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf) (practical advice for hyperparameter choices)
  - [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](https://arxiv.org/abs/1604.00788.pdf)
  - [Revisiting Character-Based Neural Machine Translation with Capacity and Compression](https://arxiv.org/pdf/1808.09943.pdf)

- [Final Projects: Custom and Default; Practical Tips](08_Practical_Tips.md)
  - [Practical Methodology](https://www.deeplearningbook.org/contents/guidelines.html) (*Deep Learning* book chapter)
  - [Model Pruning](https://papers.nips.cc/paper/2020/file/eae15aabaa768ae4a5993a8a4f4fa6e4-Paper.pdf) (Movement Pruning: Adaptive Sparsity by Fine-Tuning)
  - [Model Quantization](https://arxiv.org/pdf/2004.07320.pdf) (TRAINING WITH QUANTIZATION NOISE FOR EXTREME MODEL COMPRESSION)
  - [BabyAI](https://arxiv.org/pdf/2007.12770.pdf) (BabyAI 1.1)
  - [gSCAN](https://arxiv.org/pdf/2003.05161.pdf) (A Benchmark for Systematic Generalization in rounded Language Understanding)

- [Transformers](09_Transformer.md)
  - [Attention Is All You Need](https://arxiv.org/abs/1706.03762.pdf)
  - [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
  - [Image Transformer](https://arxiv.org/pdf/1802.05751.pdf)
  - [Music Transformer: Generating music with long-term structure](https://arxiv.org/pdf/1809.04281.pdf)

- [More about Transformers and Pretraining](10_Pretraining.md)
  - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
  - [Contextual Word Representations: A Contextual Introduction](https://arxiv.org/abs/1902.06006.pdf)

- [Question Answering](11_Question_Answering.md)
  - [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/pdf/1606.05250.pdf)
  - [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/pdf/1611.01603.pdf)
  - [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/pdf/1704.00051.pdf)
  - [Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/pdf/1906.00300.pdf)
  - [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906.pdf)
  - [Learning Dense Representations of Phrases at Scale](https://arxiv.org/pdf/2012.12624.pdf)

- [Natural Language Generation](12_Natural_Language_Generation.md)
  - [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751.pdf)
  - [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368.pdf)
  - [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833.pdf)
  - [How NOT To Evaluate Your Dialogue System](https://arxiv.org/abs/1603.08023.pdf)

- Reference in Language and Coreference Resolution
  - [Coreference Resolution chapter of Jurafsky and Martin](https://web.stanford.edu/~jurafsky/slp3/22.pdf)
  - [End-to-end Neural Coreference Resolution](https://arxiv.org/pdf/1707.07045.pdf).

- T5 and large language models: The good, the bad, and the ugly
  - [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://colinraffel.com/publications/jmlr2020exploring.pdf)

- Integrating knowledge in language models
  - [ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/pdf/1905.07129.pdf)
  - [Barack’s Wife Hillary: Using Knowledge Graphs for Fact-Aware Language Modeling](https://arxiv.org/pdf/1906.07241.pdf)
  - [Pretrained Encyclopedia: Weakly Supervised Knowledge-Pretrained Language Model](https://arxiv.org/pdf/1912.09637.pdf)
  - [Language Models as Knowledge Bases?](https://www.aclweb.org/anthology/D19-1250.pdf)

- Social & Ethical Considerations in NLP Systems
- Model Analysis and Explanation
- Future of NLP + Deep Learning

## Tutorial

### Code

- [Python](./Tutorial/cs224n-python-review-code-updated.ipynb)
- [PyTorch](./Tutorial/CS224N_PyTorch_Tutorial.ipynb)
- [Gensim](./Tutorial/Gensim%20word%20vector%20visualization.ipynb)

### Theory

- [Singular Value Decomposition](https://davetang.org/file/Singular_Value_Decomposition_Tutorial.pdf) 


## Assignment

- [Assignment 1](./Assignment/A1/exploring_word_vectors.ipynb)
- [Assignment 2](./Assignment/A2)
- [Assignment 3](./Assignment/A3/a3.pdf)
- [Assignment 4](./Assignment/A4/a4.pdf)
- [Assignment 5](./Assignment/A5/a5.pdf)







