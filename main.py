"""
Write a code in Python:
1. Load DSP_6.csv.
2. Fill (or remove) all missing data.
3. Use "Dummy coding" (sex and embark variables in DSP_6.csv).
4. Run logistic regression (test sieze = 0.1; random state = 101; y = Survived).
5. Use the same code (2.-4.), but load DSP_2 instead (y = HeartDisease)

Provide answer to following questions:
1. How can you describe the first model? Is it good?
2. How can you describe the second model? Is it good?
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load DSP_6.csv.
df6 = pd.read_csv("DSP_6.csv", sep=",")

pd.set_option('display.width', None)
# 2. Fill (or remove) all missing data.
df6_removed = df6.dropna()
print("Filled DSP_6.csv: \n", df6_removed)

# 3. Use "Dummy coding" (sex and embark variables in DSP_6.csv).

df6_dummies = pd.get_dummies(df6_removed, columns=["Sex", "Embarked"])
print()
print("Dummy coding for \"Sex\" and \"Embark\" for DSP_6: \n", df6_dummies)
#  Dummy coding is a method of reformatting data to make it work better with mathematical models, for machine
#  learning algorithms for example. It splits data like this: For column "Sex" it will replace it with
#  all variations found in the data: "Sex_male" and "Sex_female", then it will assign 0 or 1 for every
#  record based on which value was present in column "Sex"

# 4. Run logistic regression (test size = 0.1; random state = 101; y = Survived).
X = df6_dummies.drop('Survived', axis=1)  # saves all values except for Survived
y = df6_dummies['Survived']  # saved the values of the column Survived

# split data, test_size means that 90% of data goes into training and 10% into testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

model = LogisticRegression(random_state=101)  # creating instance of Logistic Regression

# dropping all irrelevant tables as they are irrelevant and also produce errors because of strings
# I realized later that I could've done it before assigning X and y but whatever, at least it will be clear for my future self
X_train_processed = X_train.drop(["PassengerId", "Pclass", "Name", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin"], axis=1)
X_test_processed = X_test.drop(["PassengerId", "Pclass", "Name", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin"], axis=1)

model.fit(X_train_processed, y_train) # training the model
y_pred = model.predict(X_test_processed) # creating the predictions

print(classification_report(y_test, y_pred, target_names=["died", "survived"])) # passing the information about the real results and then the predictions of the model
# Explanation as to what this even produces(more of a self-note):
# There are terms like "True positive"(TP), "False positive"(FP), "True negative"(TN) and "False negative". In the context
# of binary classification where one class is labeled as positive and the other as negative "false negatives" would refer to cases where
# model incorrectly predicted a negative outcome when the true label was positive
# HOWEVER, the meaning of this will change depending on context: If we are talking about "recall" for the negative class the
# "false negative" will actually mean cases which were actually positive, but our model falsely predicted them as negative
# so just be careful with this, it's mind-bending

# Precision - The ratio of correctly predicted positive observations to the total predicted positive observations. High precision means
# that the model makes few false positive predictions

# Recall - Ratio of true positives to the sum of true positives and false negatives. In other words: The ratio of our model's correct guesses to
# the sum of our model's correct guesses and our model's incorrect guesses for this category. To explain better:
# If we have 5 cases where the person died and our model predicted 3 of them correctly:
# Recall = 3 / (3+2) where 2 represents the number of cases where our model predicted survival but the person actually died

# F1-score - harmonic mean of Precision and Recall

# Support - the number of samples in each class



# 5. Use the same code (2.-4.), but load DSP_2 instead (y = HeartDisease)
print()
print("HERE ARE THE RESULTS PRINTED FOR DSP_2 NOT DSP_6 ANYMORE")
print()

# 1. Load DSP_2.csv.
df2 = pd.read_csv("DSP_2.csv", sep=",")

# 2. Fill (or remove) all missing data.
df2_removed = df2.dropna()

# 3. Use "Dummy coding" (sex and embark variables in DSP_6.csv).

df2_dummies = pd.get_dummies(df2_removed, columns=["Sex", "Embarked"])  # TODO fix this

# 4. Run logistic regression (test size = 0.1; random state = 101; y = Survived).
X = df2_dummies.drop('Survived', axis=1)  # saves all values except for Survived
y = df2_dummies['Survived']  # saved the values of the column Survived

# split data, test_size means that 90% of data goes into training and 10% into testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

model = LogisticRegression(random_state=101)  # creating instance of Logistic Regression

# dropping all irrelevant tables as they are irrelevant and also produce errors because of strings
# I realized later that I could've done it before assigning X and y but whatever, at least it will be clear for my future self
X_train_processed = X_train.drop(["PassengerId", "Pclass", "Name", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin"], axis=1)
X_test_processed = X_test.drop(["PassengerId", "Pclass", "Name", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin"], axis=1)

model.fit(X_train_processed, y_train) # training the model
y_pred = model.predict(X_test_processed) # creating the predictions

print(classification_report(y_test, y_pred, target_names=["died", "survived"]))


















