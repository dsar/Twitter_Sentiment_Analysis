# Twitter Sentiment Analysis

In this project, we present a comprehensive study of sentiment analysis on Twitter data, where the task is to predict the smiley to be positive or negative, given the tweet message.
With a fully automated framework, we developed and experimented with the most powerful proposed solutions in the related literature, including text preprocessing, text representation, also known as feature extraction, and supervised classification techniques.
Different combinations of these algorithms led to a better understanding of each component and exhausting test procedures resulted in a very high classification score on our final results.

## Project Specification

See [Project Specification](https://github.com/epfml/ML_course/tree/master/projects/project2/project_text_classification) at EPFL [/epfml/ML_course](https://github.com/epfml/ML_course) github page.

## Dependencies

In order to run the project you will need the following dependencies installed:

### Libraries

* [Anaconda3] - Download and install Anaconda with python3
* [Scikit-Learn] - Download scikit-learn library with conda

    ```sh
    $ conda install scikit-learn
    ```

* [Gensim] - Install Gensim library 

    ```sh
    $ conda install gensim
    ```
    
* [NLTK] - Download all packages of NLTK

    ```sh
    $ python
    $ >>> import nltk
    $ >>> nltk.download()
    ```

    and then download all packages from the GUI

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
    (Recommended version **tensorflow 0.12.0**)

### Files
* Stanford Pretrained Glove Word Embeddings

    Download [Glove Pretrained Word Embeddings](http://nlp.stanford.edu/data/glove.twitter.27B.zip).
    Then, unzip the downloaded file and move the extracted files in `data/glove/` directory.
    The default Data is the 200d (cp **glove.twitter.27B.200d.txt** into `data/glove/` directory).

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
    - a **Unix-based** Operating System (e.g Linux, OSX). Tested & Developed on Ubuntu

## Kaggle Submission

See the [Public Leaderboard](https://inclass.kaggle.com/c/epfml-text/leaderboard) in Kaggle.

Our Team's name is **gLove is in the air...**♩ ♪ ♫ ♬:heart:

## Demo

Go to `src/` directory and set `algorithms` variable in `options.py` file.

In case you want to parametrise the model's parameters, just set the corresponding
dictionary in `options.py`

*For more details, check the important parameters in each algorithm in the aforementioned file.*

Then just start `main.py` file

```sh
$ cd src/
$ python main.py
```

When the program terminates, you will get all the predictions of the test file
in `data/submissions/` directory

*By enabling the `cv` option to true in the `options.py` file (in the corresponding algorithm) you can get
a good approximation of the kaggle-score directly from Cross Validation (this might take a while for the full datasets)*

### Reproduce Best Score

In `data/models/BEST` directory, we have stored a checkpoint of our best CNN trained model.

Go to `options.py` file and:

* Set `algorithm` to `CNN`
* and on `CNN` dictionary:
    * set `train` to `False`
    * and make sure `checkpoint_dir` is set to `TF_SAVE_PATH + '/BEST/checkpoints'`

Finally, just follow the Demo procedure.

### Contributors

- Beril Besbinar
- Dimitrios Sarigiannis
- Panayiotis Smeros


   [Anaconda3]: <https://www.continuum.io/downloads>
   [Scikit-Learn]: <http://scikit-learn.org/stable/install.html>
   [Gensim]: <https://radimrehurek.com/gensim/>
   [NLTK]: <http://www.nltk.org/>
   [GloVe]: <https://github.com/maciejkula/glove-python>
   [FastText]: <https://pypi.python.org/pypi/fasttext>
   [Tensorflow]: <https://www.tensorflow.org/get_started/os_setup>
   

___

License: [MIT](https://opensource.org/licenses/MIT)
