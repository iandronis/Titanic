### Titanic: Machine Learning from Disaster (Kaggle Competition)
A quick implementation of the Kaggle competition with a simple
feature extraction and some basic classifiers.

#### About the competition
> The sinking of the RMS Titanic is one of the most infamous shipwrecks
> in history.  On April 15, 1912, during her maiden voyage, the Titanic
> sank after colliding with an iceberg, killing 1502 out of 2224
> passengers and crew. This sensational tragedy shocked the
> international community and led to better safety regulations
> for ships.
>
> One of the reasons that the shipwreck led to such loss of life was
> that there were not enough lifeboats for the passengers and crew.
> Although there was some element of luck involved in surviving the
> sinking, some groups of people were more likely to survive than
> others, such as women, children, and the upper-class.
>
> In this challenge, we ask you to complete the analysis of what sorts
> of people were likely to survive. In particular, we ask you to apply
> the tools of machine learning to predict which passengers survived
> the tragedy.

From the competition [homepage](https://www.kaggle.com/c/titanic).

#### Data review
The competition gives you two datasets (for training and testing).
Each dataset consists of instances (in this case passengers) with
some more information.

To be more specific, for each passenger it gives us:
* PassengerId: passenger id (ex. 2)
* Pclass: ticket class (ex. 1)
* Name: passenger's name (ex. 'Cumings, Mrs. John Bradley (Florence
Briggs Thayer)')
* Sex: passenger's sex (ex. female)
* Age: passenger's age (ex. 38)
* SibSp: number of siblings / spouses aboard the Titanic (ex. 1)
* Parch: number of parents / children aboard the Titanic (ex. 0)
* Ticket: ticket number (ex. PC 17599)
* Fare: passenger fare (ex. 71.2833)
* Cabin: cabin number (ex. C85)
* Embarked: port of embarkation (ex. C)

It also gives us the correct labels (i.e. if the passenger survives
or not) for the training dataset.

#### Feature extraction
The features I am using are:
* the Pclass
* the Name (categorize them in ['Mr.', 'Mrs.', 'Miss.', 'Master.'])
* the Sex (categorize them in ['male', 'female'])
* the Age (categorize them in [age<16, 16<age<39, 39<age<55, 55<age])
* the SibSp
* the Parch
* the Fare (categorize them in [age<30, 30<age<94, 94<age])
* the Cabin (categorize them in
['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
* the Embarked (categorize them in ['C', 'Q', 'S'])

In order to find the limits I use for the ages and fares, I draw some
plots for the train instances.

#### Classifiers
The classifiers I used are:
* Dummy Classifier
* K-Neighbors Classifier (where K = 5)
* Bernoulli Naive Bayes
* Multinomial Naive Bayes
* Gaussian Naive Bayes
* Logistic Regression (where C = 1e5)

#### Final classification and Evaluation
According to the learning curves and precision-recall curves, I decided
to use Logistic Regression (with C = 1e5) which gives me a 78.47% score.
