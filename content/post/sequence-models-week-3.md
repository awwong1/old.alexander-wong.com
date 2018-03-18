---
title: "Sequence Models, Week 3"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2018-03-17T12:27:35-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Sequence Models** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

{{% toc %}}

# Sequence models & Attention mechanism

- Sequence models can have an attention mechanism.
- Algorithm will help your model understand where it should focus its attention given a sequence of inputs.
- This week, we will cover speech recognition and how to deal with audio data.

## Various sequence to sequence architectures

### Basic Models

This week, we will cover neural network models for handing sequence input to sequence output. (Machine translation, for instance)
  - Neural network is made of two parts, an encoder network for the sequence input which outputs a vector
  - A decoder network which takes the vector as input and outputs a sequence as output

![sequence_to_sequence_model](/img/deeplearning-ai/sequence_to_sequence_model.png)

Also effective for image captioning. If the goal is to input an image and output a caption, it is possible to to take an existing architecture (like AlexNet) and instead of outputting the values to a softmax layer, one can feed this to an RNN which generates the caption one word at at time,

![image_captioning](/img/deeplearning-ai/image_captioning.png)

There are some differences between this and generating sequences (last week).

### Picking the most likely sentence

Machine translation is building a conditional language model. It requires two different networks instead of a single recurrent neural network.

There is an encoder network which figures out some representation of the input sentence before attempting to generate the translated output sequence. It is trying to predict the probability of an input sentence (in English, for example), conditioned by the probability of an output sentence (in French, for example).

![machine_translation_as_conditional_model](/img/deeplearning-ai/machine_translation_as_conditional_model.png)

How do you find the mosy likely translation for a given sentence? You don't want to sample at random. Want to find the english sentence $y$ that maximizes the translation likeliness probability.

![finding_most_likely_translation](/img/deeplearning-ai/finding_most_likely_translation.png)

Most common way to find the best sentence is to use an algorithm called `Beam search`. But why not just take a greedy appraoch (take words by most likeliness)?

![why_not_greedy_search](/img/deeplearning-ai/why_not_greedy_search.png)

Greedy search does poorly on maximizing the joint likeliness, because it focuses word as a time. We need to focus on the entire sequence as a whole.

### Beam search

Continuing with our sample input sentence "Jane visite l'Afrique en septembre", how do we search for the best possible English translation? (simplification, all words in lower case)

![beam_search_step_1](/img/deeplearning-ai/beam_search_step_1.png)

1. Evalue the first most likely word. Beamsearch has a parameter called $B = \text{beam width}$ which allows it to choose three choices for each sequence.

![beam_search_step_2](/img/deeplearning-ai/beam_search_step_2.png)

2. For each choice, consider what the next word is. Evaluate the probablity of the next word. Find likeliness of first and second word as a pair.

$$ P(y^{<1>}, y^{<2>} | x) = P(y^{<1>} | x) * P(y^{<2>}|x, y^{<1>}) $$

Take only the $B$ most likely options. At every step, you instantiate three copies of your network with different choices for your words.

![beam_search_step_3](/img/deeplearning-ai/beam_search_step_3.png)

3. Continue for the rest of the sequence, chaining all the prior words as conditioning, keeping only the top $B$ choices at each step of the sequence. This will be terminated by the `<EOS>` symbol.

If the beam width is set to be one $B=1$, this is essentially the greedy search algorithm.

### Refinements to Beam Search

![beam_length_normalization](/img/deeplearning-ai/beam_length_normalization.png)

We are trying to find the probability of an output sequence $y$, given an input sequence $x$. This can be expressed as:

$$ P(y^{\<1>}, \dots, y^{\<t\_y>} | x) = P(y^{<1>}|x) * P(y^{<2>}|x,y^{<1>}) * \dots * P(y^{\<t\_y>}|x, y^{\<1>}, \dots , y^{\<t\_y - 1>}) $$
$$ P(y^{<1>}, \dots, y^{\<t\_y>}|x) = \arg\max\_y\prod\limits\_{t=1}^{T\_y}P(y^{\<t>}|x, y^{\<1>}, \dots, y^{\<t-1>}) $$

When implementing this, most of these probabilities are much less than one. $ P(y|x) \approx 0$

In practice, we typically take the logs (this is a more numerically stable algorithm). The log function makes the search less prone to numerical underflow. Also, rather than multiplying a lot of numbers that are less than one (which is much more likely to give you a very small value), you take the sum of all logs of the probabilities (less likely to give you a very small value).

$$ \arg\max\_y\sum\limits\_{t=1}^{T\_y}\log P(y^{\<t>}|x, y^{\<1>}, \dots, y^{\<t-1>}) $$

Because summing the logs may also result in a very small value, an additional trick is to normalize by the output sequence length. This reduces the penalty for longer sentence translations.

$$ \dfrac{1}{T\_y^\alpha}\sum\limits\_{t=1}^{T\_y}\log P(y^{\<t>}|x, y^{\<1>}, \dots, y^{\<t-1>}) $$
$$ \alpha \approx 0.7 $$

How to choose the beam width $B$? The larger the value, the more choices, but also the more computationally expensive.

If the beam width is very large, you consider a lot of possibilities. Memory requrements also grow.

If the beam width is very small, then you get a worse result however this is faster.

In practice, Beam width is usually around ~10. A width of ~100 is very large for a production system. A beam width of >1000 is usually for milking a publication.

Try out different values of $B$ until you start to see diminishing returns.

Unlike exact search algorithms BFS, DFS, etc. Beam Search runs faster but is not guaranteed to find the exact maximum for $\arg\max\_y P(y|x)$.

### Error analysis in Beam Search

![beam_search_error_analysis](/img/deeplearning-ai/beam_search_error_analysis.png)

Beam search is an approximate search algorithm. How to check if beam search is making mistakes, or if the RNN model is causing mistakes?

Continuing with the example sentence: 
$$x = \text{Jane visite l'Afrique en septembre.} $$

Human translation returns:
$$ y^* = \text{Jane visits Africa in September.} $$
Algorithm returns: 
$$ \hat{y} = \text{Jane visited Africa last September.} $$

The algorithm returns a bad translation because it changes the meaning of the sentence. There are two components of the model, your neural network encoder/decoder, and your beam search. How do you know which part of your model made the error?

Your RNN computes $P(y|x)$. To determine error, you should use your RNN to compute $P(y^*|x)$ and $P(\hat{y}|x)$. The value with the higher probability indicates which portion of your learning algorithm needs fine tuning.

![error_analysis_to_beam_search](/img/deeplearning-ai/error_analysis_to_beam_search.png)

- If $ P(y^*|x) \gt P(\hat{y}|x) $, then:
  - Beam search is improperly choosing the highest probability sequence
- Else $ P(y^*|x) \leq P(\hat{y}|x) $
  - RNN predicted that the poor translation was a better candidate than the correct translation and needs additional training/tuning.

![beam_search_error_analysis_process](/img/deeplearning-ai/beam_search_error_analysis_process.png)

Going through your dev set, you can determine which part of your algorithm to focus your attention on.

### Bleu Score

![evaluating_machine_translation](/img/deeplearning-ai/evaluating_machine_translation.png)

Given an input sentence, there could be multiple translations that are equally as plausible. How do you evaluate this? You evalue this by using the Blue Score.

Example input: "Le chat est sur le tapis".

Outputs:

- The cat is on the mat.
- There is a cat on the mat.

Bleu score measure how good the generated translation is. Bleu stands for Bilingual evaluation understudy.

Given an isolated word, count how many times it appears in the generated translation sentence, then count how many times it appears in your human translations. (Take the highest count of all human translation inputs as your numerator, take the count of all the machine translation values as your denominator.)

![bleu_score_on_bigrams](/img/deeplearning-ai/bleu_score_on_bigrams.png)

Find all sequential pairs of words (bigrams) in the machine translation output, then calculate the same fraction. Take the sum of all the bigrams found in the clipped human translations and divide it over the total unique bigrams found in the machine translation output.

![bleu_score_on_unigrams](/img/deeplearning-ai/bleu_score_on_unigrams.png)

The generic formula for your $n\text{-gram}$ is:

$$ P\_n = \dfrac{ \sum\limits\_{n\text{-gram} \in \hat{y}} \text{Count}\_{\text{Clip}}(n\text{-gram}) }{ \sum\limits\_{n\text{-gram} \in \hat{y}} \text{Count}(n\text{-gram}) } $$

If the machine translation matches the human translation, all $P\_n$ values will be equal to $1$.

![bleu_score_details](/img/deeplearning-ai/bleu_score_details.png)

The final Bleu score (if you only care for up to 4 sequences) is calculated by:

$$ \text{BP} \exp (\dfrac{1}{4} \sum\limits\_{n=1}^4 P\_n) $$
$$ \text{BP} = \text{ Brevity Penalty} $$

BP tries to normalize for shorter translations. The naive bleu score will favor shorter translations, so BP will set the score lower if the machine generated translation length is less than the human translation length.

This is useful for handling any comparison between sequences.

### Attention Model Intuition

So far, we have used an encoder/decoder type network for handling a task such as text translation. Attention is an improvement to this architecture. Consider the problem of long sequences.

![the_problem_of_long_sequences](/img/deeplearning-ai/the_problem_of_long_sequences.png)

A human would not memorize the entire input sentence before beginning the output. Rather, it is more likely that the human would translate the sentence part by part as they progress through the sentence structure.

![attention_model_intuition](/img/deeplearning-ai/attention_model_intuition.png)

Using a bi-directional RNN, attempt to perform machine translation for a given sentence. There is a hidden recurrent neural network that takes the output of the first bi-directional RNN. It has takes as input context $c$ which is generated by alpha variables $\alpha^{\<1,1>}, \alpha^{\<1, 2>}, \dots, \alpha^{\<t, t'>}$ to create the output sequence.

### Attention Model

![attention_model](/img/deeplearning-ai/attention_model.png)

Assume you have an input sentence, and you use a bi-directional RNN (GRU, LSTM) to generate a set of activations (for that timestep).

$$ a^{\<t'>} = ( \overrightarrow{a}^{\<t'\>}, \overleftarrow{a}^{\<t'\>} ) $$ 

This is fed into a single-direction RNN as context $c$, where $c$ is the weighted sum of alphas from the lower bi-direction RNN.

$$ c^{\<1>} = \sum\limits\_{t'} \alpha^{\<1,t'>}a^{\<t'>} $$
$$ \alpha^{\<t, t'>} = \text{amount of 'attention' that } y^{\<t>} \text{ should pay to } a^{\<t'>} $$

![computing_attention](/img/deeplearning-ai/computing_attention.png)

$$ a^{\<t, t'>} = \dfrac{ \exp(e^{\<t,t'>}) }{\sum\limits\_{t'=1}^{T\_x}\exp(e^{\<t,t'>}) } $$

^ This formula ensures that all of your attention weights sum to 1. (ensures that your sequence is evaluated equally)

How do you compute $e$? You can use a small neural network and trust backpropagation.

$$ \text{Neural network state from previous time step} = s^{\<t-1>} $$
$$ \text{Output of current state of Bi-Directional NN} = a^{\<t'>} $$

One downside is this algorithm takes quadratic cost to run. If you have $T\_x$ as your input sequence length and $T\_y$ as your output sequence length, then your total number of attention parameters is $T\_x * T\_y$. In typical machine translation, the input and output is usually not that long, such that perhaps quadratic cost is acceptable.

## Speech recognition - Audio data

### Speech Recognition

![speech_recognition_problem](/img/deeplearning-ai/speech_recognition_problem.png)

Sequence to sequence models can perform very accurate speech recognition. Speech recognition problem is to take an audio clip $x$ and automatically find a text transcript $y$.

$x$ is air pressure over time. $y$ is a sequence of words. A common pre-processing step is to run your audio clip and generate a spectrogram.

Once upon a time, speech recognition algorithms used to be generated using phonemes. Linguists used to hypothesize that phonemes were the building blocks of all sounds. However, with end to end deep learning, phonemes are no longer necessary (no longer require hand engineered representations of audio).

Datasets of transcribed audio can range between 300 hours to 3000 hours, etc. Best commercial systems are trained on over 100,000 hours of audio.

![ctc_cost_for_speech_recognition](/img/deeplearning-ai/ctc_cost_for_speech_recognition.png)

One algorithm for speech recognition can be modeled with a 1000 to 1000 sequence neural network by collapsing repeating charcters.

Today, building a production scale speech recognition system requires a huge training set. Trigger word detection requires less data to train an effective learning algorithm.

### Trigger Word Detection

![trigger_word_detection_examples](/img/deeplearning-ai/trigger_word_detection_examples.png)

There have been more and more devices that 'wake up' on input voice.

![trigger_word_detection_algorithm](/img/deeplearning-ai/trigger_word_detection_algorithm.png)

Literature is still evolving, but this is one example of an algorithm that can be used. Use a recurrent neural network. Input data is audio (maybe preprocessed as a spectrogram), the training data $y$ is set to 1 after the trigger word is spoken and 0 otherwise. (If the training data is too imbalanced, you can pad the time after trigger word is said with more 1's)

## Conclusion

You made it! Few final thoughts:

1. Neural Networks and Deep Learning
2. Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization
3. Structuring Machine Learning Projects
4. Convolutional Neural Networks
5. Sequence Models

Deep learning is a 'super power'. With deep learning, you can make a computer see, synthesize art, generate music, translate language, render powerful models from medical sensor inputs.

Do whatever you think is the best you can do for humanity.

![udia](/img/touch-icon-apple.png)
