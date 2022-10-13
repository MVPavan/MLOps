# library doc string


# import libraries
import shap
import joblib
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report
import os
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'

images_path = Path(__file__).parent/"images"
images_path.mkdir(parents=True, exist_ok=True)

def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    return pd.read_csv(pth)

def plots(df_column, plot_type):
    """
    df_column: series to plot
    plot_type: ["hist","bar","sns_hist_density"]
    return: matplotlib figure
    """
    plt.figure(figsize=(20,10)) 
    if plot_type=="hist":
        fig = df_column.hist()
    if plot_type=="bar":
        fig = df_column.plot(kind="bar")
    if plot_type=="sns_hist_density":
        # fig = sns.histplot(df_column, stat='density', kde=True)
        fig = sns.distplot(df_column, kde=True)
    if plot_type=="sns_heatmap":
        fig = sns.heatmap(df_column.corr(), annot=False, cmap='Dark2_r', linewidths = 2)

    return fig.get_figure()
        

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    plots(
        df_column=df["Churn"], plot_type="hist"
    ).savefig(images_path/"Churn_Histogram.png")
    plots(
        df_column=df["Customer_Age"], plot_type="hist"
    ).savefig(images_path/"Customer_Age_Histogram.png")
    plots(
        df_column=df.Marital_Status.value_counts('normalize'), plot_type="bar"
    ).savefig(images_path/"Marital_Status_Bar.png")
    plots(
        df_column=df['Total_Trans_Ct'], plot_type="sns_hist_density"
    ).savefig(images_path/"Total_Trans_Ct.png")
    plots(
        df_column=df, plot_type="sns_heatmap"
    ).savefig(images_path/"Heatmap.png")
    
    


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass