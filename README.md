# Naive-Bayesian-Classifier-Airplanes-or-Birds

Implementation of a Naïve Bayesian classifier that calculates and reports the probability that 10 unidentified objects belongs to one of two classes: airplanes and birds.

## Title

CS 131 HW 05 - Naive Bayesian Classification

## Author

Brandon Dionisio

## Purpose

Using a Naïve Recursive Bayesian classifier, calculate and report the
probability that 10 unidentified objects belongs to one of two classes.

## The likelihood of birds and airplanes for specified speeds

The likelihood distribution for airplanes is in the first row of likelihood.txt and the likelihood distribution for birds is in the second row of likelihood.txt

![image](https://github.com/brandondionisio/Naive-Bayesian-Classifier-Airplanes-or-Birds/assets/145251710/60611cec-63f0-4e1a-88a0-8cdd284a91a9)

## Training Data

Ten tracks representing the velocity of the unidentified flying object measured by a military-grade radar (1s sampling frequency for a total length of 600s) in training.txt

![image](https://github.com/brandondionisio/Naive-Bayesian-Classifier-Airplanes-or-Birds/assets/145251710/0dbed7d9-5655-4838-ae8f-881607d06e38)

If the radar could not acquire the target and perform the measurement, the corresponding data point is a NaN. These tracks are raw data.

## Testing Data

Twenty tracks representing the velocity of the birds and airplanes (10 rows of birds followed by 10 rows of airplanes in training.txt) measured by a military-grade radar (1s sampling frequency for a total length of 600s). If the radar could not acquire the target and perform the measurement, the corresponding data point is an NaN value. These tracks are curated to have a maximum sample drop rate of 5% of the total number of samples per track.

## Acknowledgements

stackoverflow  
CS 131 Canvas Slides  
scikit-learn.org

## Running The Program

To run the radar, use "python radar.py"

## User Inputs

None

## Features

Feature 1: Likelihood distribution of the class based on speed as given
by likelihood.txt

Feature 2: Likelihood distribution of the class based on variance. This
distribution was divided up as the variance of speeds every 6 seconds.

## Preprocessing data

To preprocess the training data and the testing data, all NaN values are
turned into the mean for each row.

## Additional notes

To prevent 0 probabilities in the likelihood distributions, I added 0.001
to each probability. This would prevent any invalid predictions or
divisions by 0. 

In normalizing the data, I divided the combined
probabilities for both classes by 2.4 as this is the total probability
from the addition of 0.001 for both features.
