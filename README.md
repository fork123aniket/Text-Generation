# Text Generation
This repository demonstrates the word-level language modelling using LSTM (Long Short-Term Memory) networks in Tensorflow and Keras. The LSTM model is trained to predict next word in the sequence by capturing semantic inherent context of the words it has seen in the sequence during training time. The trained LSTM model can then be used to generate any number of words as a new text sequence that share similar statistical properties as the original training sequence data.
## Requirements
- `Python 3.9`
- `tensorflow 2.8`
- `numpy 1.22.3`
## Usage
- `load_doc()` loads the data and `clean_doc()` preprocesses the data to make it suitable for training the LSTM model. Moreover, `save_doc()` is used to save the data to the desired path attributed to `out_filename` variable.
- `build_model()` builds the model.
The average loss and associated model accuracy are printed after every epoch. Furthermore, to train a new network, run `Text Generation.py`. To extend the model to use pre-trained GloVe embeddings, please set `pre_trained` variable to `True`. Besides, all hyperparameters to control training and generation of the new text sequences are provided in the given `.py` file.
### Generation
- `generate_seq()` uses the training dataset to randomly sample the line (set of words) from it and then create vocabulary dictionary for the words lie in the randomly sampled sequence.
- `generate_seq()` then generates the words using the preTrained LSTM-based language model.
- `n_words` variable can be set to generate as many words as intended to form a new text sequence.
## Samples
### Input text sequence from *The Republic by Plato* dataset
```
when he said that a man when he grows old may learn many things for he can no more learn much than he 
can run much youth is the time for any extraordinary toil of course and therefore calculation and 
geometry and all the other elements of instruction which are a
```
### Generated text sequence
```
preparation for dialectic should be presented to the name of idle spendthrifts of whom the other is the 
manifold and the unjust and is the best and the other which delighted to be the opening of the soul of 
the soul and the embroiderer will have to be said at
```
