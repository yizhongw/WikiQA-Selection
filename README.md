# WikiQA-Rank
Answer selection task based on WikiQA datatest.

## Dataset

You can find the dataset at [WikiQA](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/)

I remove the question-answer groups without any correct answers in test data.

The raw data I use is published at `data/raw/` folder.

For CNN model, I use pre-trained [Glove word embeddings](http://nlp.stanford.edu/projects/glove/). You should download it and put it in `data/embeddings/` folder.

## Task

Given a question and a group of answer candidates, you should rank the candidates by the likelihood that it is the correct answer.

The quality of the rank is evaluated by MRR and MAP, as implemented in `eval.py`


## Workflow

1. Use preprocess.py to lemmatize the corpus and generate transformed train data.

2. Test two baseline methods with baseline.py:
    
    * Word matching: 
    
        ```python3 baseline.py --word_matching && python3 eval.py data/output/WikiQA-dev.rank data/raw/WikiQA-dev.tsv```
    
    * Do nothing: 
    
        ```python3 baseline.py --nothing && python3 eval.py data/output/WikiQA-dev.rank data/raw/WikiQA-dev.tsv```
    
3. Try CNN models using main.py:

    * Prepare data_helper: 
    
        ```python3 main.py --prepare```
    
    * Train cnn model: 
    
        ```python3 main.py --train```
    
    * Generate the final rank for test data: 
    
        ```python3 main.py --test```
        
    * Visualize the train loss and graph:
    
        ```python3 -m tensorflow.tensorboard --logdir data/model/summary/```
    
    
    
