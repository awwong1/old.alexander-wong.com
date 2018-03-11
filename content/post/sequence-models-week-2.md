---
title: "Sequence Models, Week 2"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2018-03-10T12:27:35-07:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Sequence Models** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

{{% toc %}}

# Natural Langauge Processing & Word Embeddings

- Learn about how to use deep learning for natraul language processing.
- Use word vector representations and embedding layers to train recurrent neural networks with great performance.
- Learn to perform sentiment analysis, named entity recognition, machine translation.

## Introduction to Word Embeddings

### Word Representation

- So far, words have been represented with a 1-hot vector of a word vocabular list.
- One of the weaknesses of this representation is it treats each word as a thing of it self. It's difficult to generalize across different words.
- The inner product between two one-hot vectors is zero.

![word_representation](/img/deeplearning-ai/word_representation.png)

It would be better if each word could be represented by features.

For instance, a word could have a gender associated with it.

The word `man` could have gender `-1` and the word `woman` could have gender `+1` while a word like `apple` could have gender `0`.

![featurized_representation_word_embedding](/img/deeplearning-ai/featurized_representation_word_embedding.png)

Notation is $e\_5391$ where the subscript value is the original one hot vector index, but $e$ is referring to the feature vector instead of the one-hot vector.

It's common to visualize word embeddings in a 2D plane (using an algorithm like t-SNE). These are called embeddings, as each word is applied to a point in a multi-dimensional space (each point has it's own space). T-SNE allows you to visualize this in a lower diemsntional space.

![visualizing_word_embeddings](/img/deeplearning-ai/visualizing_word_embeddings.png)

### Using word embeddings

**Named entity recognition**, trying to detect people's names in a sentence.

![named_entity_recognition_example](/img/deeplearning-ai/named_entity_recognition_example.png)

Word embeddings can look at 1B to 100B words. (this is common)
Training set can be around 100K words.

This knowledge can be transferred to named entity recognition, as you can train your neural network's word embeddings on text found on the internet.

1. Learn word embeddings from a large text corpus (1-100B words). Or you can downloda pre-trained embedding online
2. Transfer embedding to new task with a smaller training set (say 100k words.) Rather than using a 10,000 one hot vector, you can now use a 300 dimension dense vector.
3. Optional: Continue to fine tune the word embeddings with new data.

Finally, word embeddings have an interesting relationship to face encoding.For face recognition, recall the siamese network training for generating encoding for the input image's face. The image encoding is similar to the word embedding, however word embeddings usually have a fixed size dictionary and a unknown variable.

![relation_to_face_encoding](/img/deeplearning-ai/relation_to_face_encoding.png)

### Properties of word embeddings

Suppose that you are given the question "Man is to woman, as King is to ?"

Is it possible to have the neural network answer this question using the word embeddings? Yes it is. One interesting property of word embeddings is that the you can subtract the vector $e\_{\text{man}} - e\_{\text{woman}}$ you can compare that to the vector $e\_{\text{king}} - e\_{\text{queen}}$.

![word_embedding_ananalogies](/img/deeplearning-ai/word_embedding_ananalogies.png)

Your algorithm can first subtract the vectors man/woman to calculate the difference for finding a similar analogy for king to ?.

In pictures, perhaps the word embedding is in 300 dimensional space. The vector difference between Man and Woman is very similar to the vector difference between King and Queen. The arrow difference in the slide below represents a difference in gender.

Try to find the word $w$ such that the following equation holds true.
$$ e\_{\text{man}} - e\_{\text{woman}} \approx e\_{\text{king}} - e\_{\text{queen}}$$

Find word w: $ \text{arg} \max\limits\_w \text{similarity}(e\_w, e\_{\text{king}} - e\_{\text{man}} + e\_{\text{woman}})$

If you learn a set of word embeddings, you can find analogies using word vectors with decent accuracy.

![analogies_using_word_vectors](/img/deeplearning-ai/analogies_using_word_vectors.png)

The most commonly used similarity function is **Cosine similarity**.

$$\text{similarity}(u, v) = \dfrac{u^Tv}{||u||\_2||v||\_2} $$

Can also use euclidian distance. $ ||u-v||^2 $.

Things it can learn:

- Man:Woman as Boy:Girl
- Ottawa:Canada as Nairobi:Kenya
- Big:Bigger as Tall:Taller
- Yen:Japan as Ruble:Russia

### Embedding matrix

When you implement an algorithm to learn an embedding, you're learning an embedding matrix.

Take for instance, a 10,000 word vocaulary. `[a, aaron, orange, ... zulu, <UNK>]`

The let's make this matrix E = 300 by 10,000. If Orange was indexed 6257,

$O\_{6257}$ is the one hot vector of 10,000 rows with a 1 at the 6257th position.

![embedding_matrix](/img/deeplearning-ai/embedding_matrix.png)

$$ E * O\_j = e\_j $$

Initialize E randomlly, then use gradient descent to learn the parameters of the embedding matrix.

In practice, you use a specialized function to do the multiplication, as matrix multiplication is innefficient with many one hot vectors. Keras has an embedding module that does this for you.

## Learning Word Embeddings: Word2vec & GloVe

### Learning word embeddings

Lets say you're building a language model. Building a neural language model is a reasonable way to learn word embeddings.

![neural_language_model](/img/deeplearning-ai/neural_language_model.png)

You can have various context and target pairs.

![other_context_target_pairs](/img/deeplearning-ai/other_context_target_pairs.png)

### Word2Vec

Recall the above **Skip-grams**.
Let's say you're given the sentence "I want a glass of orange juice to go along with my cereal". Rather than having context be the immediate last word, randomly pick a word to be your context word. Then, randomly pick a word within your window (plus or minus four words) to be your target word.

![skip-grams](/img/deeplearning-ai/skip-grams.png)

Model:

- Vocab size = 10,000 words
- Want to learn a mapping from some context c ("orange") to some target t ("juice")

![word2vec_model](/img/deeplearning-ai/word2vec_model.png)

Softmax : $$ p(t|c) = \dfrac{e^{\theta\_t^Te\_c}}{\sum\limits\_{j=1}^{10,000}e^{\theta\_j^Te\_c}} $$

$$ \theta\_t = \text{parameter associated with output t} $$

Loss function:
$$ \mathcal{L}(\hat{y}, y) = - \sum\limits\_{i=1}^{10,000} y\_i \log \hat{y}\_i $$

Problems with softmax classification:

- You need to carry out a sum over your entire vocabulary every time you want to calculate a probability.
   - use a hiearchical softmax classifier. Think of it as a decision tree with binary/logistic classifier. This scales with log of vocablary size, rather than linear scale with vocablary size.

![problems_with_softmax_classification](/img/deeplearning-ai/problems_with_softmax_classification.png)

In practice, $ P( c)$ is not taken uniformly randomly. There are issues of getting a lot of common words 'and, or, to' etc. We choose words more likely to result in a better embedding matrix.

### Negative Sampling

The downside of the last step is the softmax step is slow to compute. This algorithm is much more efficient.

Define a new learning problem

- I want a glass of orange juice to go along with my cereal.

Given a pair of words, orange:juice, determien if it is a context target pair.
orange:juice returns `1`, while orange:king returns `0`.

![define_a_new_learning_problem](/img/deeplearning-ai/define_a_new_learning_problem.png)

Pick a valid context/word pair, then pick a bunch of random variablse from the dictionary and then set them to be `0` (random words are usually not content linked.)

Define a logistic regression model.

$$ P(y=1 | c, t)  = \sigma (\theta\_t^Te\_c) $$

![negative_sampling_model](/img/deeplearning-ai/negative_sampling_model.png)

You are only updating K=1 binary classification probelms rather than updating a 10,000 array. This is called neagtive sampling because you have a positive example, yet you go out and generate a bunch of negative examples afterwards.

How do you choose the negative examples? After choosing the context word "orange", how do you choose the negative examples?
- One thing you can do is sample the candidate target words according to the imperial frequency of words in your corpus. (how often it appears) The problem is it gives you a bunch of words like "The, of, and, ..."

Imperically, what they though to work best:

$$ p(w\_i) = \dfrac{f(w\_i)^{3 / 4}}{\sum\limits\_{j=1}^{10000} f(w\_j)^{3 / 4}} $$ 

![selecting_negative_examples](/img/deeplearning-ai/selecting_negative_examples.png)

### GloVe word vectors

GloVe stands for "global vectors for word representation".

"I want a glass of orange juice to go along with my cereal."

$$ x\_{ij} = \text{ # times i appears in context of j} $$

- How often do words appear close with each other?

![global_vectors_for_word_representation](/img/deeplearning-ai/global_vectors_for_word_representation.png)

![glove_model](/img/deeplearning-ai/glove_model.png)

Minimize:

$$ \sum\limits\_{i=1}^{10,000} \sum\limits\_{j=1}^{10,000} f(X\_{ij})(\Theta\_i^Te\_j + b\_i + b\_j' - \log{X\_{ij}})^2 $$
$$ f(X\_{ij}) = 0 \text{if} X\_{ij} = 0 $$
- $f$ accounts for $0\log{0} = 0 $.
- $f$ also accounts for frequent words (this, is, of, a) and infrequent words (durian)

**Note on the featurization view of word embeddings**

- features learned using these algorithms do not neatly translate to interperatable features like 'gender', or 'royal'.

![note_on_word_embeddings](/img/deeplearning-ai/note_on_word_embeddings.png)

## Applications using Word Embeddings

### Sentiment Classification

Task of looking at a piece of text, telling if the text is "liked" or "disliked".

![sentiment_classification_problem](/img/deeplearning-ai/sentiment_classification_problem.png)

- You may not have a huge labeled dataset.

**Simple sentiment classification model**

![simple_sentiment_classification_model](/img/deeplearning-ai/simple_sentiment_classification_model.png)

- Take your words as one hot vectors, multiply it by the embedding matrix to extract out your word's embedding vector
- Averaging your embedding vectors, then take the softmax classifier's output as your value
- cons: this ignores word order

**RNN for sentiment classification**

![rnn_for_sentiment_classification](/img/deeplearning-ai/rnn_for_sentiment_classification.png)

- Many to One architecture RNN that takes in your entire sequence and output a softmax output

### Debiasing word embeddings

How to deminish/eliminate bias (gender, race, etc.) in word embeddings.

![the_problem_of_bias_in_word_embeddings](/img/deeplearning-ai/the_problem_of_bias_in_word_embeddings.png)

- Man:Woman as King:Queen
- Man:Computer_Programmer as Woman:Homemaker (probably not right)
  - Man:Computer_Programmer as Woman:Computer Programmer
- Father:Doctor as Mother:Nurse (although Doctor would have been better)

The biases picked up reflects the biases written by people. Difficult to scrub when you train on a lot of historical data.

![addressing_bias_in_word_embeddings](/img/deeplearning-ai/addressing_bias_in_word_embeddings.png)

1. Identify the bias direction.
2. Neutralize: For every word that is not definitional, project to get rid of bias.
3. Equalize pairs.
- Authors trained a classifier to determine which words were definitional and which words were not definitional. This helped detect which words to neutralize (to project out bias direction).
- Number of pairs to equalize is usually very small.
