import numpy as np
import pandas as pd
titanic = pd.read_csv("/Users/nehansh/Documents/MachineLearning_Class/Assignment_2/train.csv")
titanic_test = pd.read_csv("/Users/nehansh/Documents/MachineLearning_Class/Assignment_2/test.csv")
#cleansing
for df in [titanic, titanic_test]:
    df.loc[df["Sex"] == "male", "Sex"] = 0
    df.loc[df["Sex"] == "female", "Sex"] = 1
    # df["Embarked"] = df["Embarked"].fillna("S")
    # ports = list(df["Embarked"].unique())
    # for i,port in enumerate(ports):
    #     df.loc[df["Embarked"]==port, "Embarked"] = i
    for col in ["Age", "Fare"]:
        df[col] = df[col].fillna(titanic[col].median())
    print(df.describe())
