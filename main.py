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
print("Dummy coding for \"Sex\" and \"Embarked\" for DSP_6: \n", df6_dummies)
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

print("HERE ARE THE RESULTS PRINTED FOR DSP_6:")
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
print("HERE ARE THE RESULTS PRINTED FOR DSP_2:")

# 1.
df2 = pd.read_csv("DSP_2.csv", sep=",")

# 2.
df2_removed = df2.dropna()[["Sex", "ChestPainType", "RestingECG", "ST_Slope", "HeartDisease"]]

# 3.
# I have chosen columns: Sex, ChestPainType, RestingECG and ST_Slope because they don't have many unique values like Age or Cholesterol,
# which the model we are using would struggle with as it considers every unique value separately, not making a connection between for example
# 283 and 287 levels of cholesterol despite them being close to each other. To use such columns we would have to use a different method or at
# least reformat the data into repeatable values like "average", "high" etc.
df2_dummies = pd.get_dummies(df2_removed, columns=["Sex", "ChestPainType", "RestingECG", "ST_Slope"])

# 4.
X_df2 = df2_dummies.drop('HeartDisease', axis=1)  # saves all values except for HeartDisease
y_df2 = df2_dummies['HeartDisease']  # saved the values of the column HeartDisease

X_train_df2, X_test_df2, y_train_df2, y_test_df2 = train_test_split(X_df2, y_df2, test_size=0.1, random_state=101)

model_df2 = LogisticRegression(random_state=101)

model_df2.fit(X_train_df2, y_train_df2)
y_pred_df2 = model_df2.predict(X_test_df2)

print(classification_report(y_test_df2, y_pred_df2, target_names=["no heart disease", "did experience a heart disease"]))


"""
Provide answer to following questions:
1. How can you describe the first model? Is it good?
2. How can you describe the second model? Is it good?
"""

# My answer to these questions:
# 1. I would say that the first model is of rather low quality. While it is significantly better at predicting survival rather than death, even that
# doesn't have a great degree of accuracy. Overall I would say that the model lacks training and testing data to allow it to perform better
# and us to get a more definitive results

# 2. I would say that because of the significantly higher sample size(over 900 records after removal - none were removed actually - compared to 183 the
# Titanic data had) the precision of this model has been much better. We have achieved relatively high precision and recall values and very consistent
# averages across all provided types. While the model could definitely be improved with more data, perhaps letting us reach accuracy of above 90%, as
# it currently stands the model performed pretty well