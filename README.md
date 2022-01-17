# BERT-Bi-LSTM-Bot-Recognition

├───.ipynb_checkpoints
├───An-Empirical-study-on-Pre-trained-Embeddings-and-Language-Models-for-Bot-Detection [1], [2]
│   └───dataset generation
│       └───classification_processed
├───BERT+BiLSTM_SSB 
├───BERT+BiLSTM_Scraped
├───BERT_SCRAPED
├───BERT_SSB
├───Baselines
├───Datasets
│   └───datasets_full
│       └───ssb_tokenized
└───text-preprocessing-techniques [3]

This work has borrowed preprocessing techniques and datasets from Garcia-Silva et al. [1], Gilani et al. [2] and Kamps et al. [3]. 

**An-Empirical-study-on-Pre-trained-Embeddings-and-Language-Models-for-Bot-Detection** is a cloned folder from the project of Garcia-Silva et al. to scrape and preprocess a dataset.

The following folders contain experiments with all the datasets

BERT+BiLSTM_SSB 
├───BERT+BiLSTM_Scraped
├───BERT_SCRAPED
├───BERT_SSB
├───Baselines

**text-preprocessing-techniques** Contains a custom preprocessing from Kamps et al.

Utilites.py and Utilities2.py contain sampling and dataset generation scripts.

Preprocessing_test_set.ipynb samples the test set of the dataset SSB2 and SSB3

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[1] A. Garcia-Silva, C. Berrio, and J. M. Gómez-Pérez, “An Empirical Study on Pre-trained Embeddings and Language Models for Bot Detection,” in Proceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP-2019), 2019, pp. 148– 155, doi: 10.18653/v1/W19-4317 [Online]. Available: https://www.aclweb.org/anthology/W19-4317. [Accessed: 14-Jan 2022]

[2] Z. Gilani, E. Kochmar, and J. Crowcroft, “Classification of Twitter Accounts into Automated Agents and Human Users,” in Proceedings of the 2017 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining 2017, 2017, pp. 489–496, doi: 10.1145/3110025.3110091 [Online]. Available: https://dl.acm.org/doi/10.1145/3110025.3110091. [Accessed: 15-Jan-2022]

[22] D. Effrosynidis, S. Symeonidis, and A. Arampatzis, “A Comparison of Pre-processing Techniques for Twitter Sentiment Analysis,” in Research and Advanced Technology for Digital Libraries, vol. 10450, J. Kamps, G. Tsakonas, Y. Manolopoulos, L. Iliadis, and I. Karydis, Eds. Cham: Springer International Publishing, 2017, pp. 394–406 [Online]. Available: http://link.springer.com/10.1007/978-3- 319-67008-9_31. [Accessed: 16-Jan-2022]
