```markdown
# Natural Language Processing

## Introduction

Natural Language Processing (NLP) is the field of Artificial Intelligence that focuses on the interaction between computers and humans using the natural language. The goal of NLP is to enable computers to understand, interpret, and generate human languages in a valuable way. This notebook covers the fundamental aspects of NLP, including text preprocessing, word embeddings, sequence models, transformer architectures, and practical applications.

## Text Preprocessing

### Theory

Text preprocessing is the initial phase in NLP tasks which involves cleaning and transforming raw text data into a format that is understandable and analyzable for models. Key steps include:

1. **Tokenization**: Splitting text into words or sentences.
2. **Stemming and Lemmatization**: Reducing words to their base or root form.
3. **Stop Word Removal**: Removing common words that don't contribute much to the meaning (e.g., "is", "the").
4. **Lowercasing**: Converting all text to lower case to ensure uniformity.
5. **Removing Punctuation and Special Characters**: Cleaning text by removing any irrelevant symbols.

Mathematically, tokenization can be represented as a function `f(t)`, where `t` is a string, and `f` divides `t` into a sequence of words or tokens.

### Examples

Python provides libraries like `nltk` and `spaCy` for text preprocessing:

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Sample text
text = "Natural Language Processing is a complex field."

# Tokenization
tokens = word_tokenize(text.lower())

# Stop words removal
filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]

print("Original:", text)
print("Stemmed:", stemmed_words)

```

## Word Embeddings

### Theory

Word embeddings are dense vector representations of words in a continuous vector space where similar words have similar representations. Models like Word2Vec, GloVe, and fastText learn these representations using contexts of words.

For instance, Word2Vec uses the Continuous Bag of Words (CBOW) and Skip-Gram models. These methods aim to maximize the probability of predicting a word based on its context.

Given a vocabulary `V`, Word2Vec assigns each word `w` a vector representation and learns embeddings by minimizing the negative log-likelihood of observed (context, word) pairs:

\[ J(\theta) = -\sum \log P(w_t | context(w_t)) \]

### Examples

Let's use `gensim` library to create Word2Vec embeddings:

```python
from gensim.models import Word2Vec

# Example corpus
sentences = [
    ['natural', 'language', 'processing', 'nlp'],
    ['deep', 'learning', 'machine', 'learning', 'ai']
]

# Training Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Word Embedding for 'nlp'
print("Word Embedding for 'nlp':", model.wv['nlp'])
```

## Sequence Models

### Theory

Sequence models are designed to handle sequential data, capturing temporal dependencies and understanding patterns over sequences. Two primary models include:

- **Recurrent Neural Networks (RNNs)**: These are neural networks with loops, allowing information to persist. However, RNNs can suffer from vanishing gradient problems.

- **Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU)**: Both are designed to overcome RNN limitations by incorporating gates that manage the flow of information.

Mathematically, an LSTM cell is described by the following equations:

- Forget Gate: \( f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \)
- Input Gate: \( i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \)
- Output Gate: \( o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \)

Each layer involves these weights and biases to manage the flow of information through time.

### Examples

Using `Keras`, you can build an LSTM model as follows:

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128))
model.add(LSTM(units=64))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

## Transformer Architectures

### Theory

Transformers are the state-of-the-art in NLP due to their ability to capture long-range dependencies using self-attention mechanisms and parallelization. Unlike RNNs, transformers do not use recurrence; instead, they rely on attention mechanisms. The key components are:

- **Self-Attention**: Determines the attention score by comparing each word with every other word in a sequence.
- **Encoder-Decoder Structure**: The Transformer model comprises an encoder to process the input sequence and a decoder to generate outputs.

The attention mechanism is mathematically defined as a query `Q`, a key `K`, and a value `V` matrix, where:

\[ \text{Attention(Q,K,V)} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]

### Examples

Implementing a transformer model in `PyTorch`:

```python
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

# Example usage
simple_transformer = SimpleTransformer()
print(simple_transformer)
```

## Applications in NLP

### Theory

NLP encompasses diverse applications that transform how machines and humans interact. These include:

- **Machine Translation**: Automatically translating text or speech from one language to another.
- **Sentiment Analysis**: Determining the sentiment or emotion behind text data.
- **Named Entity Recognition (NER)**: Identifying and classifying key elements in text into predefined categories.

### Examples

An example of sentiment analysis using `Hugging Face` `transformers` library:

```python
from transformers import pipeline

# Initialize sentiment analysis pipeline
nlp_pipeline = pipeline('sentiment-analysis')

# Analyze sentiment in sample text
result = nlp_pipeline("Natural Language Processing is incredible!")

print("Sentiment Analysis Result:", result)
```

By employing these techniques and models, we can achieve impressive outcomes in various domains, enhancing the ability of machines to communicate with humans intelligently.
```
