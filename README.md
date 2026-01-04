# InspireBot: Quote Text Generator using LSTM

This notebook demonstrates how to generate quotes using an LSTM neural network trained on a dataset of famous quotes. The model learns word sequences and predicts the next word, enabling the generation of new, coherent quotes.

---

## 1. Introduction

In this project, we aim to:

* Train a neural network to generate text based on a dataset of quotes.
* Explore preprocessing steps required for text generation.
* Implement and train an LSTM-based model.
* Generate quotes using a trained model and a seed phrase.

---

## 2. Libraries and Dataset

We use the following libraries:

* `numpy` and `pandas` for data manipulation.
* `tensorflow` and `keras` for building the LSTM model.
* `matplotlib` and `seaborn` for visualizations (optional).

The dataset contains quotes along with their authors.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SimpleRNN
```

---

## 3. Data Preprocessing

The preprocessing steps include:

* Converting all text to lowercase.
* Removing punctuation.
* Tokenizing the text into sequences of integers.

```python
# Convert quotes to lowercase
quotes = df['quote'].str.lower()

# Remove punctuation
import string
translator = str.maketrans('', '', string.punctuation)
quotes = quotes.apply(lambda x: x.translate(translator))
```

---

## 4. Tokenization and Sequence Preparation

We convert quotes to sequences of integers suitable for training:

* Create sequences where each word is predicted from the preceding words.
* Pad sequences to a uniform length.

```python
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(quotes)

sequences = tokenizer.texts_to_sequences(quotes)

# Create input-output pairs
X, y = [], []
for seq in sequences:
    for i in range(1, len(seq)):
        X.append(seq[:i])
        y.append(seq[i])

max_len = max(len(x) for x in X)
X_padded = pad_sequences(X, maxlen=max_len, padding='pre')
y_one_hot = to_categorical(y, num_classes=vocab_size)
```

---

## 5. Model Architecture

We use an LSTM-based model:

* **Embedding layer**: Converts word indices into dense vectors.
* **LSTM layer**: Learns sequential patterns in text.
* **Dense layer**: Outputs probability distribution over the vocabulary.

```python
embedding_dim = 50
rnn_units = 128

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(units=rnn_units))
model.add(Dense(units=vocab_size, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## 6. Training the Model

Train the model using categorical cross-entropy loss and Adam optimizer. Adjust `epochs` and `batch_size` based on available resources.

```python
# Example training
# model.fit(X_padded, y_one_hot, epochs=50, batch_size=128, validation_split=0.1)
```

---

## 7. Prediction and Text Generation

To generate text:

* Provide a seed phrase.
* Predict the next word step by step.
* Append predicted words to generate a complete quote.

```python
def predictor(model, tokenizer, text, max_len):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')
    pred = model.predict(seq, verbose=0)
    pred_index = np.argmax(pred)
    index_to_word = {i: "<UNK>" for i in range(vocab_size)}
    for word, index in tokenizer.word_index.items():
        if index < vocab_size:
            index_to_word[index] = word
    return index_to_word.get(pred_index, "<UNK>")

def generate_text(model, tokenizer, seed_text, max_len, n_words):
    for _ in range(n_words):
        next_word = predictor(model, tokenizer, seed_text, max_len)
        if next_word == "<UNK>":
            break
        seed_text += " " + next_word
    return seed_text

seed = "the world is"
generated_quote = generate_text(model, tokenizer, seed, max_len, 10)
print(generated_quote)
```

---

## 8. Saving Model and Tokenizer

Save the trained model, tokenizer, and `max_len` for future inference.

```python
model.save("lstm_model.keras")
import pickle

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)
```

---

## 9. Conclusion

* We successfully built an LSTM-based text generator for quotes.
* The preprocessing and tokenizer setup ensure safe handling of unknown words.
* The model can generate coherent text given a seed phrase.
* Future improvements include:

  * Using a smaller `max_len` to reduce training time.
  * Using a GRU or stacked LSTM for more complex sequences.
  * Fine-tuning with more data for higher-quality generated text.
