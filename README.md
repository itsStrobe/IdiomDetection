# Automatic Detection of Idiomatic Language.

Code for research on idiomatic detection in text via supervised and semi-supervised methods.

## Dependencies

This code is written in python. To run the code from the main repository, be sure to have the following requirements:

* Python 3.6
* Theano 0.7
* A recent version of [NumPy](http://www.numpy.org/) and [Pandas](https://pandas.pydata.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)
* [NLTK 3](http://www.nltk.org/)
* [Beautiful Soup 4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) (for Corpora Processing)
* [lxml](https://lxml.de/) (XML library to work with BS4)
* [progressbar2](https://pypi.org/project/progressbar2/) (Visual progress bar for file processing)
* [gensim](https://radimrehurek.com/gensim/) (for use of Word2Vec similarity model for Lexical Fixedness calculation)

All these dependencies should be automatically installed (save for Python 3.6) after running the following command (be sure to have [virtualenv](https://virtualenv.pypa.io/en/latest/) and [pip3](https://pip.pypa.io/en/stable/) installed):

    virtualenv -p <Python 3.6 MACRO> venv && source venv/bin/activate
    pip3 install -r requirements.txt

***Note: Some of the embeddings models require different versions of Python (e.g. Python 2.7) and Pip, so be sure to pre-install them. On execution, packages should be automatically installed with the requirements.txt files using the python 2 of the previous command.***

## Getting started

You will first need to download the [BNC-XML Corpora](http://www.natcorp.ox.ac.uk/) and the [VNC-Tokens Dataset](http://multiword.sourceforge.net/PHITE.php?sitesig=FILES&page=FILES_20_Data_Sets). Be sure to update the location of these datasets in the "CORPORA_DIR" and "TARG_DIR" parameters in the follwing program files depending on where you save them:

* CleanBNC.py
* ExtractCorpora.py
* ExtractPatternsInCorpora.py
* ExtractTargetSentences.py
* ExtractTextAndTags.py
* TrainSVMs.py

## Running Code

After all dependencies are installed, code execution should run smoothly. If you only want to replicate the paper results, execute the RunAll.sh file:

    ./RunAll.sh

This will execute the Corpora clean-up and necessary pre-processing. Followed by training the embedding models and idiom vs literal use classifiers.
Feel free to modify any of the parameters set in this file for your own purposes.  

If you only want to run particular sections of the pipeline, you may run any of the following files:

* ./PrepareData.sh
* ./TrainEmbeddings.sh
* ./FindVNICs.sh
* ./GenerateEmbeddings.sh
* ./TrainSVMs.sh
* ./GenerateSilverStandard.sh
* ./RunExperiments.sh

Modify them accordingly to your purposes.

## Result Evaluation

After running the files, several directories will contain the results from the different tests and experiments.

### VNIC-Candidates

VNIC-Candidates extracted from the selected Corpora (run './FindVNICs.sh') based on the fixedness metrics will be located in the './targets' directory

### Experiments Results

Results from all the executed experiments will be saved in the './results/' sub-directory located within the directory where the Experiments code is located.

## Works Cited

This repository used code and ideas from the following sources:

* Fixedness Metrics and CForm: [Unsupervised Type and Token Identification of Idiomatic Expressions](https://www.aclweb.org/anthology/papers/J/J09/J09-1005/), Fazly et al (2009)
* Word2Vec: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781), Mikolov et al (2013)
* Siamese CBOW: [Siamese CBOW: Optimizing Word Embeddings for Sentence Representations](https://arxiv.org/abs/1606.04640), Kenter et al (2016)
* Skip-Thoughts: [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726), Kiros et al (2015)
* ELMo: [Deep contextualized word representations](http://arxiv.org/abs/1802.05365), Peters et al (2018)

## Reference

If you found this code useful, please cite the following paper:

Jose Zavala. **"Automatic Detection of Idiomatic Language."** *(2019).*

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
