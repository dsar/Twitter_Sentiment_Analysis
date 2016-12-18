# Twitter Sentiment Analysis

## Dependencies

In order to run the project you will need the following dependencies installed:

### Libraries

* [Anaconda3] - Download and install Anaconda with python3
* [Gensim] - Install Gensim library 

    ```sh
    $ conda install gensim
    ```
    
* [NLTK] - Download all the corpora of NLTK

    ```sh
    $ python
    $ >>> import nltk
    $ >>> nltk.download()
    ```

    and then download all the corpora from the GUI

* [GloVe] - Install Glove python implementation

    ```sh
    $ pip install glove_python
    ```
    
    Sometimes `libgcc` is also needed to be installed.
    ```sh
    $ pip install libgcc
    ```

* [FastText] - Install FastText implementation

    ```sh
    $ pip install fasttext
    ```
* [Tensorflow] - Install tensorflow library

    ```sh
    $ pip install tensorflow
    ```

### Files
* Pretrained Glove Word Embeddings

    Download [Glove Pretrained Word Embeddings](http://nlp.stanford.edu/data/glove.twitter.27B.zip).
    Then, unzip the downloaded file and move the extracted files in `data/glove/` directory.
    The default Data is the 200d (cp glove.twitter.27B.200d.txt into `data/glove/` directory).

* Training Data

    Download the positive [Positive & Negative tweet files](https://inclass.kaggle.com/c/epfml-text/download/twitter-datasets.zip) in order to train the models
    and move them in `data/datasets/` directory.

* Testing Data

    Download the [Testing tweet file](https://inclass.kaggle.com/c/epfml-text/download/test_data.txt) in order to test the models in kaggle
    and move it in `data/datasets/` directory.


## Hardware Requirements

- A Computer with:
    - at least **16** GB of RAM
    - a **Graphics Card** (optional - needed for faster training in CNN solution)

## Kaggle

See the [Public Leaderboard](https://inclass.kaggle.com/c/epfml-text/leaderboard) in Kaggle.

## Demo

Go to `src/ directory` and set Algorithms variable in `options.py` file.

In case you want to parametrise the model's parameters, just set the corresponding
dictionary in `options.py`

Then just start main.py file

```sh
$ python src/main.py
```

When the program terminates you will get all the predictions of the test file
in `data/submissions/` directory
    
### Contributors

- Beril Besbinar
- Dimitrios Sarigiannis
- Panayiotis Smeros


   [Anaconda3]: <https://www.continuum.io/downloads>
   [Gensim]: <https://radimrehurek.com/gensim/>
   [NLTK]: <http://www.nltk.org/>
   [GloVe]: <https://github.com/maciejkula/glove-python>
   [FastText]: <https://pypi.python.org/pypi/fasttext>
   [Tensorflow]: <https://www.tensorflow.org/get_started/os_setup>
   
