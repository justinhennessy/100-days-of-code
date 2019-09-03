# 100 Days Of Code - Log

### Day 2: September 3, 2019

**Today's Progress**

Wow, today has been slow going, I wanted to get through at a relatively high level `Unsupervised Learning` and `Deep Learning`, only managed to get through `Unsupervised`.


`Unsupervised Learning` helps us fine previously unknown patterns in a dataset without needing a label. There are a couple of main methods that are used to do this, `Principle Component Analysis` and `K-mean Clustering`.


`Principle Component Analysis` (PCA) is a `Dimensionality Reduction` technique that helps find the most relevant features (ie variables, attributes) in a dataset. `Dimentionality Reduction` is discovering non-linear, non-local relationships in data. PCA transforms variables into a new set of variables which are a linear combination of the original variables which assists in clustering data.


`K-mean Clustering` is one of the most popular clustering techniques, grouping similar data points together. `K` being the defined number of clusters you are looking for. This method works through the data points figuring out the smallest amount of distance variance between data points to determine the right number for `K` ie number of clusters.

**Thoughts:**

Tough day today after a long day at work which started at 6am. I think I have decided to change tact. I am super keen to understand all the math under the hood but after tonight I simply dont have the maths chops yet, there is a huge amount of ground work I need to do before I can dive too deep.

So on that I will start looking at Python courses to start playing with data as soon as I can. I figure if I can get something practical up and running and use the results from the maths functions and get a good grasp of their uses then that might help me understand it better.

**Link to work:**

Youtube channels:

[Siraj Raval](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A) <- this guy is amusing to watch and really entertaining, makes complex seem easier

[3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) <- only watched one video but looks like has heaps of math knowledge ill come back to

[StatQuest with Josh Starmer](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw) <- another entertaining smart dude, maths knowledge


### Day 1: September 2, 2019

**Today's Progress**

My first day has begun with research and I suspect the first week or 2 is going to be the same.

I have been working my way through the [Machine learning for humans](https://medium.com/machine-learning-for-humans/why-machine-learning-matters-6164faf1df12) blog post (I downloaded the PDF). As I am a complete noob, I wanted to get across all the different types of learning. Today was all about supervised learning, here are some tid bits.

There are two main tasks for supervised learning:

- regression (ie how much will a house sell for)
- categorisation (ie cat or dog)

Under the umbrealla of regression there are a few methods

`linear regressions` (ordinary least square, OLS), predict a continuous numerical value, tries and predicts a value Y given a previous unseen value of X. Within this topic there was discussion of `Gradient Descent`, the goal of which is to find the minimum of a model’s loss function by iteratively getting a better and better approximation of it. Machine learning libraries like scikit-learn and TensorFlow use it in the background everywhere, so it’s worth understanding the details.


`logistic regression` a method of classification which outputs the probability of a target Y belonging to a certain class

It also discussed a concept of `overfitting`, which essentially means, the model matches the training data exactly so losses its ability to "predict" or learning for things it hasn't seen.

Two ways to combat overfitting:

1. Use more training data. The more you have, the harder it is to overfit.

2. Use regularization. Add in a penalty in the loss function for building a model.

**Thoughts:**

I have decided to take up the 100 day of coding challenge with a focus on Machine Learning.

From a very early age, AI and Machine Learning has been a fascination and something I thought would only ever be a thing of fantasy. With the barrier to entry being so low now and with an abundance of tooling to help not only learn but execute, the time is ripe.

I would like to thank Daniel Bourke and Angela Baltes (part of my Linkedin network) for their guidance and patience to help me at the very beginning of my journey, this humble student appreciates the time you have given to help me on this journey to date.

So a challenge is not one without a goal, so here goes, I am committing that in 100 days, I will have built my first custom learning model and have run a real-world experiment with it.

I have some ideas on where I might start to give myself a little more focus but I will vet these ideas first, then commit to a particular use case in the coming weeks.

**Link to work:**

[Learning how to Learn](https://www.coursera.org/learn/learning-how-to-learn?utm_term=danielbourke_learning-how-to-learn_jan2019&ranMID=40328&ranEAID=EBOQAYvGY4A&ranSiteID=EBOQAYvGY4A-EE5LayT1JI2.Iaxai1d69g&siteID=EBOQAYvGY4A-EE5LayT1JI2.Iaxai1d69g&utm_content=10&utm_medium=partners&utm_source=linkshare&utm_campaign=EBOQAYvGY4A)

[Machine learning for humans](https://medium.com/machine-learning-for-humans/why-machine-learning-matters-6164faf1df12)


