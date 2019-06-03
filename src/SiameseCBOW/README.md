# Siamese CBOW

This repository contains the code of Siamese CBOW which can be used to generate word embeddings, optimized for sentence representations. If you are interested in pretrained embeddings, scroll down to the bottom. 

## Overview

Siamese CBOW is a neural network architecture for calculating word embeddings
optimised for being averaged across sentences to produce sentence
representations.

Siamese CBOW compares sentence vectors to each other.
A sentence can be a *positive example*, which means that it is supposed to be
semantically similar to the sentence it is compared to, or a
*negative example*, when it is not assumed to be semantically similar.

![Siamese CBOW architecture](img/siamese-cbow.png "Siamese CBOW architecture")

Above is a picture of the Siamese CBOW architecture.
A sentence embedding is obtained by averaging the word embeddings of the words
in a sentence.
The cosine similarity between sentence pairs is calculated, and the network
tries to give a high cosine similarity to positive examples, and a low cosine
similarity to negative examples.
Please note that there is a more elaborate picture of the architecture, that follows the actual implementation more closely, in the img/ directory.


If you use Siamese CBOW and publish about your work, please cite [Siamese CBOW: Optimizing Word Embeddings for Sentence Representations, T. Kenter, A. Borisov, M. de Rijke, ACL 2016](http://arxiv.org/pdf/1606.04640v1.pdf):

    @inproceedings{kenter2016siamesecbow,
      title={Siamese CBOW: Optimizing Word Embeddings for Sentence Representations},
      author={Kenter, Tom and Borisov, Alexey and de Rijke, Maarten},
      booktitle={Proceedings of the The 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016)},
      year={2016},
    }

### Dependencies

* [Theano](http://www.deeplearning.net/software/theano/)
* [Lasagne](http://lasagne.readthedocs.io/)
* [Gensim](http://radimrehurek.com/gensim/)

## Usage

Several input formats are supported, each of them corresponding to a slightly
different use.
In general, Siamese CBOW is called like:

    $ python siamese-cbow.py [OPTIONS] DATA OUTPUT_DIR 

### Required arguments


Argument  | 
-- | --
DATA | File (in PPDB case) or directory (in Toronto Book Corpus and INEX case) to read the data from. NOTE that the program runs in aparticular input mode (INEX/PPDB/TORONTO) which is deduced from the directory/file name)
OUTPUT_DIR |  A file to store the final and possibly intermediate word embeddings to (in cPickle format)


### Optional arguments

Argument  | 
-batch_size INT | Batch size. Default: 1
-dont_lowercase | By default, all input text is lowercased. Use this option to prevent this.
-dry_run | Build the network, print some statistics (if -v is on) and quit before training starts.
-embedding_size INT | Dimensionality of the word embeddings. Default: 300
-epochs INT | Maximum number of epochs for training. Default: 10
-h, --help | show this help message and exit
-gradient_clipping_bound FLOAT | Gradient clipping bound (so gradients will be clipped to [-FLOAT, +FLOAT]).
-last_layer LAYER | Last layer is 'cosine' or 'sigmoid'. NOTE that this choice also determines the loss function (binary cross entropy or negative sampling loss, respectively). Default: cosine
-learning_rate FLOAT | Learning rate. Default: 1.0
-max_nr_of_tokens INT |  Maximum number of tokens considered per sentence. Default: 50
-max_nr_of_vocab_words INT | Maximum number of words considered. If this is not specified, all words are considered
-momentum FLOAT | Momentum, only applies when 'nesterov' is used as update method (see -update). Default: 0.0
-neg INT | Number of negative examples. Default: 1
-share_weights | Turn this option on (a good idea in general) for the embedding weights of the input sentences and the other sentences to be shared.
-start_storing_at INT |  Start storing embeddings at epoch number INT. Default: 0. I.e. start storing right away (if -store_at_epoch is on, that is)
-store_at_batch INT | Store embeddings every INT batches.
-store_at_epoch INT | Store embeddings every INT epochs (so 1 for storing at the end of every epoch, 10 for for storing every 10 epochs, etc.).
-regularize | Use l2 normalization on the parameters of the network
-update UPDATE_ALGORITHM | Update algorithm. Options are 'adadelta', 'adamax', 'nesterov' (which uses momentum) and 'sgd'. Default: 'adadelta'
-v | Be verbose
-vocab FILE | A vocabulary file is simply a file, SORTED BY FREQUENCY of frequence<SPACE>word lines. You can take the top n of these (which is why it should be sorted by frequency). See -max_nr_of_vocab_words.
-vv | Be very verbose
-w2v FILE | A word2vec model can be used to initialize the weights for words in the vocabulary file from (missing words just get a random embedding). If the weights are not initialized this way, they will be trained from scratch.

### Data formats

Three data formats are supported: PPDB, INEX and Toronto Book Corpus.

#### PPDB

The PPDB corpus is a corpus of paired short phrases which are explicitely
marked for semantic similarity.
The corpus can be downloaded from <http://www.cis.upenn.edu/~ccb/ppdb/>.

To run Siamese CBOW on the XL version of the corpus, first construct a vocabulary file (see the section 'Preprocessing PPDB data' below), and then run this command:

    $ THEANO_FLAGS=floatX=float32 python siamese-cbow.py -v -share_weights \
      -vocab /path/to/ppdbVocabFile.txt -epochs 5 -neg 2 -embedding_size 100 \
      /path/to/PPDB/xl/ppdb-1.0-xl-phrasal myPpdbOutputDir

Running this command will results in a file with embeddings being written to
`myPpdbOutputDir/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_5_batch_1_neg_2_voc_37532x100_noReg_lc_noPreInit.pickle`.

We can inspect these embeddings by loading them in Python:

    >>> import wordEmbeddings as we
    >>> oWE_PPDB = we.wordEmbeddings("./myPpdbOutputDir/cosine_sharedWeights_adadelta_lr_1_epochs_5_batch_100_neg_2_300d_noReg_lc_noPreInit_vocab_65535.end_of_epoch_5.pickle")

    >>> oWE_PPDB_epoch_5.most_similar('bad')
    [(u'evil', 0.80611455), (u'naughty', 0.80259472), (u'mess', 0.78487486), (u'mal', 0.77596927), (u'unhappy', 0.75018817), (u'silly', 0.74343985), (u'wrong', 0.74152184), (u'dirty', 0.73908687), (u'pissed', 0.73823059), (u'terrible', 0.73761278)]
    >>> oWE_PPDB_epoch_5.most_similar('good')
    [(u'nice', 0.8981635), (u'genuinely', 0.88940138), (u'truly', 0.88519835), (u'good-looking', 0.87889814), (u'swell', 0.87747467), (u'delicious', 0.87535363), (u'adorable', 0.8678599), (u'charming', 0.86676568), (u'lovely', 0.86469257), (u'rightly', 0.8637867)]
    >>> oWE_PPDB_epoch_5.most_similar('controls')
    [(u'controlling', 0.97573024), (u'inspections', 0.97478831), (u'checks', 0.97122586), (u'surveillance', 0.96790159), (u'oversight', 0.96275693), (u'supervision', 0.95519221), (u'supervises', 0.95428753), (u'supervise', 0.95306516), (u'monitoring', 0.95126694), (u'monitors', 0.9501)]
    >>> oWE_PPDB_epoch_5.most_similar('love')
    [(u'loves', 0.92758971), (u'likes', 0.9195503), (u'apologize', 0.91565502), (u'adore', 0.89775938), (u'apologise', 0.89251494), (u'oh', 0.89192468), (u'ah', 0.88041437), (u'aw', 0.87821764), (u'dear', 0.8773343), (u'wow', 0.87290406)]

##### Preprocessing PPDB data

To construct a vocabulary file to be used by Siamese CBOW, go to the Siamese
CBOW install directory, and say:

    $ cd ./ppdbutils/
    $ ./makeVocab.sh /path/to/PPDB/xl/ppdb-1.0-xl-phrasal > path/to/ppdbVocabFile.txt

Note that this may take a while.

#### INEX

The INEX dataset is in fact a Wikipedia dump from November 2012.
It was released as a test collection for the INEX 2013 tweet contextualisation track (_Report on INEX 2013_, P. Bellot, et al, 2013).

The dataset can be downloaded from <http://inex.mmci.uni-saarland.de/tracks/qa/> but please note that a password is required (which is available to INEX participants).
If the dataset can not be obtained, probably any other Wikipedia dump will do just as well.

In order to run Siamese CBOW on INEX, we first need to preprocess the data and generate a vocabulary file form it (see the section 'Preprocessing INEX data' below).

For now, let's assume the pre-processed data is in `/path/to/inex_paragraph_output_dir/` and we have vocabulary `INEX.vocab/txt`.

Now we can call:

    $ THEANO_FLAGS=floatX=float32 python siamese-cbow.py -v -vocab INEX.vocab.txt \
     -max_nr_of_vocab_words 65535 -share_weights -epochs 2 -batch_size 100 \
     -neg 2 /path/to/inex_paragraph_output_dir/ word_embeddings_output/

We can again inspect the results:

    >>> oWE_INEX = we.wordEmbeddings("word_embeddings_ouput/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle")
    >>> oWE_INEX.most_similar('guitar')
    [(u'guitarist', 0.97509861), (u'drummer', 0.96972936), (u'vocalist', 0.9674294), (u'bands', 0.96368152), (u'label', 0.96355927), (u'recording', 0.96105915), (u'records', 0.96105659), (u'concert', 0.95835334), (u'guitars', 0.95771766), (u'recordings', 0.95764017)]
    >>> oWE_INEX.most_similar('guitarist')
    [(u'drummer', 0.99089122), (u'album', 0.98755884), (u"band's", 0.98562163), (u'albums', 0.98432165), (u'ep', 0.98385787), (u'band', 0.98266292), (u'bassist', 0.98139161), (u'vocals', 0.98104805), (u'recording', 0.98002404), (u'vocalist', 0.97645795)]
    >>> oWE_INEX.most_similar('architect')
    [(u'architects', 0.98361444), (u'architectural', 0.96871454), (u'facade', 0.96360624), (u'architecture', 0.95200473), (u'fa\xe7ade', 0.94916135), (u'brick', 0.94499725), (u'renovation', 0.94063747), (u'renovated', 0.93654388), (u'demolished', 0.93343973), (u"building's", 0.929977)]
    
##### Preprocessing INEX data

The preprocessing consists of parsing the Wikipedia XML, tokenizing the text with NLTK `tokenizers/punkt/english.pickle`, replacing some utf-8 characters, normalizing the amounts of spaces, etc.

Also, we want make use of the structure in Wikipedia, in particular the paragraph bounds, by only considering sentences to be positive examples if they are next to eachother AND are within the same paragraph.
Therefore, when we parse the Wikipedia XML, we keep track of paragraphg bounds.
We assume that by doing things this way, more semantic coherence is captured.
The downside is, however, that there are many 1- and 2-sentence paragraphs, all of which are ignored. 

Preprocessing is done by running the following script:

    $ cd /path/to/inexutils
    $ ./tokenizeInexFiles.sh /path/to/INEXQAcorpus2013 "*" \
       /path/to/inex_paragraph_output_directory 5 -paragraph_mode

The penultimate argument (5 in the example above) controls the number of processes to run in parallel.

For a further explanation of arguments, run `tokenizeInexFiles.sh` without any arguments.

The INEX Wikipedia dump comes as one file per Wikipedia page.
All these files are stored in 1000 directories.
The original directory structure of the Wikipedia dump is maintained in the output directory.

##### Generating INEX vocabulary 

NOTE that you should __run this on the pre-processed files__ (see step above).

We first make a vocabulary for every __pre-processed__ file:

    $ ./makeAllVocabs.sh 5 /path/to/inex_paragraph_output_directory /path/to/dir_with_all_vocabularies

Again, the 5 indicates that 5 processes will (at most) be running simultaneously.

We now have one directory with very many vocabulary file.
We can merge them all to one big file by saying:

    $ python combineVocabs.py /path/to/dir_with_all_vocabularies INEX.vocab.txt

#### Toronto Book Corpus

The Toronto Book Corpus contains 74,004,228 already pre-processed sentences in total, which are made up of 1,057,070,918 tokens, originating from 7,087 unique books (_Aligning books and movies: Towards story-like visual explanations by watching movies and reading books._, Zhu et al., 2015).

The corpus can be downloaded from <http://www.cs.toronto.edu/~mbweb/>.

As no paragraphs boundaries are present in this corpus, we simply treat all triples of consecutive sentences as positive examples (NOTE that these may even cross book boundaries, though very rarely of course).

For efficient handling of the Toronto Book Corpus (and to get rid of some UTF-8 encoding issues), some pre-processing needs to be done (see below)).
If we assume the pre-processing is done, and the vocabulary has been generated and is stored in `toBoCo.vocab.txt`, we can call:

    $ THEANO_FLAGS=floatX=float32 python siamese-cbow.py -v -vocab toBoCo.vocab.txt \
     -max_nr_of_vocab_words 315643 -share_weights -epochs 1 -batch_size 100 \
     -neg 2 /path/to/toBoCo_preprocessed_dir word_embeddings_output/

The vocabulary size here is chosen to allow only for words appearing 5 times or more.

##### Preprocessing Toronto Book Corpus

The Toronto Book Corpus comes in two files: `books_large_p1.txt` and `books_large_p2.txt`.
The first one contains some characters that cause utf8 encoding errors, while being of no importance.
To get rid of them, run:

    $ cd /path/to/torontobookcorpusutils
    $ python replaceWeirdChar.py /path/to/books_large_p1.txt > /path/to/books_large_p1.corrected.txt

As we need random access to these rather large files, we first compute the character offsets of every sentence in them (the files come as one sentence per line).
These character offsets enable us to extract random sentences from the corpus in an efficient way.
They are assumed to be stored in two .pickle files.
To produce these files, in the torontobookcorpusutils directory, run:

    $ python file2positions.py /path/to/toronto_book_corpus/books_large_p1.corrected.txt \
       /path/to/toronto_book_corpus/
    $ python file2positions.py /path/to/toronto_book_corpus/books_large_p2.txt \
       /path/to/toronto_book_corpus
 
Please note that, to keep things simple, the software expects the .txt files and the .pickle files to be in the same directory (as reflected in the example above).

##### Generating Toronto Book Corpus vocabulary

To generate a vocabulary file of the entire Toronto Book Corpus, run the following scripts:

    $ python makeVocab.py /path/to/vocabs/toronto_book_corpus/books_large_p1.corrected.txt > \
       books_large_p1.corrected.vocab.txt
    $ python makeVocab.py /path/to/vocabs/toronto_book_corpus/books_large_p2.txt > \
       books_large_p2.vocab.txt
    $ python ../inexutils/combineVocabs.py /path/to/vocabs toBoCo.vocab.txt

## Pretrained embeddings

To keep this repository lightweight, the pretrained embeddings are stored somewhere else:


* [Vectors trained on PPDB XL (57.4 Mb zip file, 127.1 Mb unzipped)](http://www.tomkenter.nl/siamese_cbow/pretrained_embeddings/PPDB/cosine_sharedWeights_adadelta_lr_1_epochs_5_batch_100_neg_2_300d_noReg_lc_noPreInit_vocab_65535.end_of_epoch_5.pickle.zip) These vectors are what the examples in the PPDB section above are generated from.
* [Vectors trained on the Wikipedia (INEX) corpus (100.6 Mb zip file, 223.2 Mb unzipped)](http://www.tomkenter.nl/siamese_cbow/pretrained_embeddings/INEX/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle.zip)
  These vectors are what the examples in the INEX section above are generated from.
* [Vectors trained on the Toronto Book Corpus (483.2 Mb zip file, 1.3 Gb unzipped)](http://tomkenter.nl/siamese_cbow/pretrained_embeddings/toronto_book_corpus/cosine_sharedWeights_sgd_lr_0_0001_noGradClip_epochs_5_batch_100_neg_2_voc_315644x300_noReg_lc_noPreInit_vocab_315643.end_of_epoch_1.pickle.zip)
  These are the exact vectors used to generate the numbers in 'Siamese CBOW: Optimizing Word Embeddings for Sentence Representations', Kenter et al, ACL 2016.
  What I noticed is that they are not very human-interpretable. This has something to do with the corpus, I think, as this doesn't go so much for vectors trained on other corpora (see below).