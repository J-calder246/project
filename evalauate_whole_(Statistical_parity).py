"""
Module for evaluation of the results of '2nd layer modelling' using statistical parity
    -   Statistical parity is the ratio of cases selected positive by the model compared to the favoured group
    -   Measures such as equalised odds or disparate impact cannot be used here as 'True Positives' cannot be found through the two layer moddling system.
    -   Assessment of true positive cases are assessed in the 1st layer evaluation module
"""