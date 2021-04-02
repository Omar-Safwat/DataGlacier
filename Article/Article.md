# Transfer Learning in Deep Learning

## Introduction

Transfer learning is a machine learning technique where a model trained on one task is re-purposed on a second related task.
This optimization allows improved performance when modeling the second task, given how challenging deep learning is, and how dependant it is on the amount of data used. On the other hand, it is not exclusively an area of study, yet it is a popular method in deep learning. <br>

In order for transfer learning to work, the features learned from the first task must be general, in other words, it must be the features must be suitable for both the first and the second task.<br>

There is a stark difference between the traditional approach of building and training machine learning models, and using a methodology following transfer learning principles.

<img src="traditional vs transfer.jpg">

In tradiotional learning methods specific tasks are isolated, and separated by different datasets and training. In other words, no retained knowledge is transfered from one task to another. Therefore, in transfer learning, you can leverage knowledge (features, weights, etc..) from previously trained models for training newer models.

## Why use Transfer learning

For  starters, creating labelled data is expensive, so optimally leveraging existing datasets is key. It also emphasizes the generality of your model by starting from patterns that have been learned for a different task, thus ensuring its versatility. This is especially true for NLP and Computer vision models, where a famous collections of images from [ImageNet](http://www.image-net.org/) or corpus from [WordNet](https://wordnet.princeton.edu/) are used more than often in research or even in practice, especially when data is scarce. 

Another critical benefit of transfer learning is reduced training time because it can sometimes take days or even weeks to train a deep neural network from scratch on a complex task. According to DeepMind CEO Demis Hassabis, transfer learning is also one of the most promising techniques that could lead to artificial general intelligence (AGI) someday.

## How to apply Transfer learning

There are two common approaches to transfer learning:
1. Develop Model Approach
2. Pre-trained Model Approach

### Develop Model Approach

This approach requires you to first build a model from scratch on a base dataset. The base data has to be general and must have features in common with your 2nd task. This requires you to develop a skillful model rather than a naive one.<br>
The base model is then used as a starting point for your specific task of interest, where the model is optinionally refined and tuned for your purposes.

### Pre-trained Model Approach

A pre-trained source model is chosen from available models. Many research institutions release models on large and challenging datasets that may be included in the pool of candidate models from which to choose from.

### Feature extraction

Another advantage of using Transfer learning is automatic feature extraction. This approach is also known as representation learning, and can often result in a much better performance than can be obtained with hand-designed representation.

<img src="featurization.jpg"><br>

Because in machine learning, features are usually manually hand-crafted by researchers and domain experts, deep learning can make this task alot easier by automatically extracting relevant features from a pre-trained model. A representation learning algorithm can discover a good combination of features within a very short timeframe, even for complex tasks which would otherwise require a lot of human effort. 

Essentialy, in a featuriser you use the first layers of the network to determine the useful feature, but you dont use the output of the network, as it is too task-specific. But how can we reuse existing networks for feature extraction ? Because deep learning systems and models are layered architectures that learn different features at different layers (hierarchical representations of layered features) it is possible to feed a data sample into the network, and take one of the intermediate layers in the network as output. 

<img src="feature extraction.jpg">

__So how well do these of-the-shelf-features work in practice on different tasks ?__ <br>

<img src="improvement of featurisation.jpg">

Based on the red and pink bars in the above figure, you can clearly see that the features from the pre-trained models consistently out-perform very specialized task-focused deep learning models.

## Examples of Transfer learning

### Transfer Learning with Image Data

Probably an application where transfer learning is most commonly used, taking photos or videos as input data to train the model.<br>

It is not uncommon to use pre-trained deep learning models for large classification tasks such as the ImageNet 1000-class photograph classification competition.

The research organizations that develop models for this competition and do well often release their final model under a permissive license for reuse. These models can take days or weeks to train on modern hardware.

These models can be downloaded and incorporated directly into new models that expect image data as input.

To mention a few models of this type:
* [Oxford VGG Model](https://www.robots.ox.ac.uk/~vgg/research/very_deep/)
* [Google Inception Model](https://github.com/google/inception)
* [Microsoft ResNet Model](https://github.com/KaimingHe/deep-residual-networks)

More examples can also be found at [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)

### Transfer Learning with NLP

Pre-trained models can aslo come in handy when training language models that use text as input and output, as they represent all sorts of challenges. These are usually transformed or vectorized using different techniques. Embeddings, such as Word2vec and FastText, have been prepared using different training datasets. Moreover, newer models like the Universal Sentence Encoder and BERT definitely present a myriad of possibilities for the future.

Examples of this type of models:
* [Google’s word2vec Model](https://code.google.com/archive/p/word2vec/)
* [Stanford’s GloVe Model](https://nlp.stanford.edu/projects/glove/)

## How Transfer Learning can Improve your model

Transfer learning optimizes your model and saves you time and resources. As seen in the image below, your model can reap three benefits from it:

1. __Higher start:__ The initial skill (before refining the model) on the source model is higher than it otherwise would be.
2. __Higher slope:__ The rate of improvement of skill during training of the source model is steeper than it otherwise would be.
3. __Higher asymptote:__ The converged skill of the trained model is better than it otherwise would be.
<img src="benefits of transfer learning.jpg" width=500>

## Transfer Learning Challenges

Although transfer learning as immense potention and is commonly used in practice nowadays, there are certain pertinent issues related to transfer learning that need more research and exploration.<br>

* __Negative Transfer:__ Negative transfer refers to scenarios where the transfer of knowledge from the source to the target does not lead to any improvement, but rather causes a drop in the overall performance of the target task. There can be various reasons for negative transfer, such as cases when the source task is not sufficiently related to the target task, or if the transfer method could not leverage the relationship between the source and target tasks very well.

## Summary of popular pre-trained models

One of the fundamental requirements for transfer learning is the presence of models that perform well on source tasks. Luckily, the deep learning world believes in sharing. Many of the state-of-the art deep learning architectures have been openly shared by their respective teams. These span across different domains, such as computer vision and NLP, the two most popular domains for deep learning applications.<br>

The famous deep learning Python library, keras, provides an interface to download some popular models. You can also access pre-trained models from the web since most of them have been open-sourced.

__Pre-trained models for Computer vision:__

* [VGG-16](https://www.kaggle.com/keras/vgg16/home)
* [VGG-19](https://www.kaggle.com/keras/vgg19/home)
* [Inception V3](https://www.kaggle.com/keras/vgg19/home)
* [XCeption](https://arxiv.org/abs/1610.02357)
* [ResNet-50](https://www.kaggle.com/keras/resnet50/home)

__Word embedding models for NLP:__

* [Google’s word2vec Model](https://code.google.com/archive/p/word2vec/)
* [Stanford’s GloVe Model](https://nlp.stanford.edu/projects/glove/)
* [FastText](https://fasttext.cc/)

## References

* [A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)
* [A Gentle Introduction to Transfer Learning for Deep Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/#:~:text=Transfer%20learning%20is%20a%20machine,model%20on%20a%20second%20task.&text=Common%20examples%20of%20transfer%20learning,your%20own%20predictive%20modeling%20problems.)
* [What is transfer learning? exploring the popular deep learning approach](https://builtin.com/data-science/transfer-learning)
* [Transfer Learning: Leverage Insights from Big Data](https://www.datacamp.com/community/tutorials/transfer-learning)
