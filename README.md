#  ACT CW2: Analysing the Correlation Between the Quality & Price of a Book & Whether an Author Guarantees an Audience:
This data science project is based on the ideas of critical and commercial hits and flops. The pop culture landscape with entertainment typically revolves around well known faces with companies banking on them to generate income for a given project. 
The only drawback to this method of thinking is that most productions involve hundreds to thousands of people with budgets ballooning to inlcude everyone involved to pay and credit them for their work. 

While I could have used a dataset to reflect this industry,
I chose to focus on one that sole depends on at least one singular person and at most, two people. What I wanted to study was whether an author's name was enough to guarantee an audience as well as a high price tag. To do this, I used a Kaggle dataset using metadata from Amazon books on the marketplace. There were 20,000 books available on the dataset with the majority being
from debut authors. The idea is that a neural network is trained not only on an author's name but their experience. Initially, I was going to also include the publication date but there were some date 
formatting errors.

## Structure of Project:

This project consists of three Jupyter notebooks as well as a python file containing all my functions which will be called to analyse this question using a traditional approach as well as a neural network approach to see whether an author can rely on themselves to 
generate revenue for their projects. Each notebook can be run easily by running each coding cell to see the output displayed. 

For further analysis, I will assess whether pre-training affects the performance of the neural network. This question will allow me to represent whether a person can be biased into thinkning that a given author is more successful on one storefront than another. The results obtained from this question will be compared to the initial approach in Question 2.

The dependencies that are required will be listed in the dependencies.txt file.

## Tackling the project:
Initially, I began by seeing whether there was a linear correlation between any two variables. Unfortunately, I found that the only correlations were polynomial. 
As such, I used clustering to see whether any one author would stand out among poor quality and poor prices as well has high quality and high price as well as high quality and low price.
Using a similar method, I also checked to see whether the consumers had a preferred publisher among the three clusters. 

With the neural network, I used multiple inputs that compared different aspects to check to see whether a book would be successful provided the marketing reached a general market audience. 
The idea behind this was clear: a consumer won't usually take in one variable when they decide to purchase a product. Any product must check a few internal boxes before they consume. 

Approaching the research question, I changed the batch size and epoch variables between the second and third notebook and comparing the error loss and the accuracy as well as the F1 score. 
