import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pandas.plotting import scatter_matrix

import statsmodels.api as sm


# load frame data and perform basic analysis
def analysis():
    # load frame data
    print("Loading frame data...")
    framedata = load_framedata()
    # start with linear regression
    print("Performing linear regression...")
    full_linear_regression(framedata)
    # component wise linear regression
    print("Performing component-wise linear regression...")
    partial_linear_regression(framedata, "AIR SPEED", "RUN SPEED")

def load_framedata():
    run_speed = pd.read_csv("framedata/run_speed.csv", sep="\t")
    run_speed.name = "RUN SPEED"
    run_speed.rename(columns={"MAX RUN SPEED VALUE" : "RUN SPEED"}, inplace=True)
    air_speed = pd.read_csv("framedata/air_speed.csv", sep="\t")
    air_speed.name = "AIR SPEED"
    air_speed.rename(columns={"MAX AIR SPEED VALUE" : "AIR SPEED"}, inplace=True)
    air_acceleration = pd.read_csv("framedata/air_acceleration.csv", sep="\t")
    air_acceleration.name = "AIR ACCEL"
    air_acceleration.rename(columns={"BASE VALUE" : "BASE AIR ACCEL", "MAX ADDITIONAL" : "ADDN AIR ACCEL", "TOTAL" : "TOTAL AIR ACCEL"}, inplace=True)
    walk_speed = pd.read_csv("framedata/walk_speed.csv", sep="\t")
    walk_speed.name = "WALK SPEED"
    walk_speed.rename(columns={"MAX WALK SPEED VALUE" : "WALK SPEED"}, inplace=True)
    fall_speed = pd.read_csv("framedata/fall_speed.csv", sep="\t")
    fall_speed.name = "FALL SPEED"
    fall_speed.rename(columns={"MAX FALL SPEED" : "FALL SPEED"}, inplace=True)
    initial_dash = pd.read_csv("framedata/initial_dash.csv", sep="\t")
    initial_dash.name = "INITIAL DASH"
    initial_dash.rename(columns={"INITIAL DASH VALUE" : "INITIAL DASH"}, inplace=True)
    weight = pd.read_csv("framedata/weight.csv", sep="\t")
    weight.name = "WEIGHT"
    weight.rename(columns={"WEIGHT VALUE" : "WEIGHT"}, inplace=True)
    # my personal tier list for prediction
    tierlist = pd.read_csv("framedata/tier_list.csv", sep="\t")

    # Exploratory Data Analysis 
    framedata_list = [run_speed, air_speed, air_acceleration, walk_speed, fall_speed, initial_dash, weight]

    # Remove metric rank from columns and see summary
    for metric in framedata_list:
        print("Initial metrics for {}".format(metric.name))
        new_header(metric.name + " COLUMNS")
        print(metric.columns)
        new_header(metric.name + " DESCRIBE")
        print(metric.describe())
        new_header(metric.name + " REMOVING RANK")
        metric.drop("RANK", axis=1, inplace=True)
        print(metric.head())

    # merge metrics 
    new_header("MERGING")
    merged_framedata = framedata_list[0]
    for i in range(1, len(framedata_list)):
        metric = framedata_list[i]
        merged_framedata = merged_framedata.merge(metric, on="CHARACTER")
    # merge with tier list
    merged_framedata = merged_framedata.merge(tierlist, on="CHARACTER")

    return merged_framedata

# full scatter matrix
def display_framedata(framedata):
    scatter_matrix(framedata)
    plt.show()

# most components
def full_linear_regression(framedata):
    X = framedata[["AIR SPEED", "RUN SPEED", "WEIGHT", "FALL SPEED", "TOTAL AIR ACCEL"]]
    X = sm.add_constant(X)
    y = framedata["RANK"]

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())

# two particular components
def partial_linear_regression(framedata, predictor, target):
    print("Comparing {} and {}".format(predictor, target))
    X = framedata[predictor]
    X = sm.add_constant(X)
    y = framedata[target]

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())

    scat = scatter_matrix(framedata[[predictor, target]], alpha=0.5, marker='o', s=100)
    scat[0, 0].set_yticklabels(['0.8', '1.0', '1.2'])
    plt.show()

# quick way to separate sections
def new_header(title, length=30, char="="):
    print(char*length + " " + title + " " + char*length)


if __name__ == "__main__":
    analysis()

