"""
Churn Library module
"""
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import constansts as consts

sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

images_path = Path(__file__).parent / "images"
images_path.mkdir(parents=True, exist_ok=True)

models_path = Path(__file__).parent / "models"
models_path.mkdir(parents=True, exist_ok=True)


@dataclass
class CLVariables:
    """
    Collection of Churn Library Variables
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.lfc = None
        self.rfc = None
        self.y_train_preds_rf = None
        self.y_test_preds_rf = None
        self.y_train_preds_lr = None
        self.y_test_preds_lr = None


class ChurnLibrary:
    """
    ChurnLibrary Class
    """

    def __init__(self, pth):
        '''
        creates dataframe for the csv found at pth
        input:
                pth: a path to the csv
        output:
                self.df pandas dataframe
        '''
        self.df = pd.read_csv(pth)
        self.cl_vars = CLVariables()

    @staticmethod
    def plot_func(df_column, plot_type, save_name):
        """
        df_column: series to plot
        plot_type: ["hist","bar","sns_hist_density"]
        saves matplotlib figure
        """
        fig = plt.figure(figsize=(20, 10))
        if plot_type == "hist":
            fig = df_column.hist()
        if plot_type == "bar":
            fig = df_column.plot(kind="bar")
        if plot_type == "sns_hist_density":
            # fig = sns.histplot(df_column, stat='density', kde=True)
            fig = sns.distplot(df_column, kde=True)
        if plot_type == "sns_heatmap":
            fig = sns.heatmap(
                df_column.corr(
                    numeric_only=True),
                annot=False,
                cmap='Dark2_r',
                linewidths=2)

        fig.get_figure().savefig(images_path / save_name)

    def perform_eda(self, plots=True):
        '''
        perform eda on df and save figures to images folder
        plots: Plots and saves eda graphs
        output: None
        '''
        self.df['Churn'] = self.df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        if not plots:
            return
        ChurnLibrary.plot_func(
            df_column=self.df["Churn"],
            plot_type="hist", save_name=consts.eda_plots["Churn"])
        ChurnLibrary.plot_func(
            df_column=self.df["Customer_Age"],
            plot_type="hist", save_name=consts.eda_plots["Customer_Age"])
        ChurnLibrary.plot_func(
            df_column=self.df["Marital_Status"].value_counts("normalize"),
            plot_type="bar", save_name=consts.eda_plots["Marital_Status"])
        ChurnLibrary.plot_func(
            df_column=self.df['Total_Trans_Ct'],
            plot_type="sns_hist_density",
            save_name=consts.eda_plots["Total_Trans_Ct"])
        ChurnLibrary.plot_func(
            df_column=self.df,
            plot_type="sns_heatmap", save_name=consts.eda_plots["heatmap"])

    def encoder_helper_loop(self):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        converts catergorical column to numerical/onehot encoding and appends to dataframe
        '''
        for cat_col, cat_col_new in consts.cat_columns.items():
            self.df[cat_col_new] = self.df[cat_col].map(
                self.df.groupby(cat_col).mean(numeric_only=True)['Churn'])
            consts.quant_columns.append(cat_col_new)

    def perform_feature_engineering(self):
        '''
        input:
                df: pandas dataframe
                response: string of response name [optional argument that could be
                used for naming variables or index self.cl_vars.y column]

        output:
                X_train: self.cl_vars.X training data
                X_test: self.cl_vars.X testing data
                y_train: self.cl_vars.y training data
                y_test: self.cl_vars.y testing data
        '''
        self.cl_vars.X = self.df[consts.quant_columns]
        self.cl_vars.y = self.df["Churn"]
        self.cl_vars.X_train, self.cl_vars.X_test, self.cl_vars.y_train, self.cl_vars.y_test =\
            train_test_split(self.cl_vars.X, self.cl_vars.y, test_size=0.3, random_state=42)

    def save_models(self):
        """
        Saves Models
        """
        joblib.dump(self.cl_vars.rfc, models_path / consts.model_names["rfc"])
        joblib.dump(self.cl_vars.lrc, models_path / consts.model_names["lr"])

    def load_models(self, load_pretrained=True):
        """
        Load Models
        """
        if load_pretrained:
            name_dict = consts.model_names_pretrained
        else:
            name_dict = consts.model_names
        self.cl_vars.rfc = joblib.load(models_path / name_dict["rfc"])
        self.cl_vars.lrc = joblib.load(models_path / name_dict["lrc"])

    def classification_report_image(self):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        output:
                None
        '''
        plt.clf()
        plt.rc('figure', figsize=(5, 5))
        # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
        # approach
        plt.text(0.01, 1.25, str('Random Forest Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    self.cl_vars.y_train, self.cl_vars.y_train_preds_rf)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str('Random Forest Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    self.cl_vars.y_test, self.cl_vars.y_test_preds_rf)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(images_path / consts.result_plots["rfc_accuracy"])

        plt.clf()
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Logistic Regression Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    self.cl_vars.y_train, self.cl_vars.y_train_preds_lr)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    self.cl_vars.y_test, self.cl_vars.y_test_preds_lr)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(images_path / consts.result_plots["lr_accuracy"])

    def feature_importance_plot(self):
        '''
        creates and stores the feature importances in pth
        '''
        plt.clf()
        explainer = shap.TreeExplainer(self.cl_vars.rfc)
        shap_values = explainer.shap_values(self.cl_vars.X_test)
        shap.summary_plot(
            shap_values,
            self.cl_vars.X_test,
            plot_type="bar",
            show=False)
        plt.savefig(images_path / consts.result_plots["shap"])

        # Calculate feature importances
        importances = self.cl_vars.rfc.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        # Rearrange feature names so they match the sorted feature importances
        names = [self.cl_vars.X.columns[i] for i in indices]
        # Create plot
        fig = plt.figure(figsize=(20, 5))
        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(self.cl_vars.X.shape[1]), importances[indices])
        # Add feature names as x-axis labels
        plt.xticks(range(self.cl_vars.X.shape[1]), names, rotation=90)
        fig.get_figure().savefig(images_path / consts.result_plots["features"])

    def roc_plot(self):
        '''
        creates and stores the feature importances in pth
        '''
        lrc_plot = plot_roc_curve(
            self.cl_vars.lrc,
            self.cl_vars.X_test,
            self.cl_vars.y_test)

        fig = plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(
            self.cl_vars.rfc,
            self.cl_vars.X_test,
            self.cl_vars.y_test,
            ax=ax,
            alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        fig.get_figure().savefig(images_path / consts.result_plots["roc"])

    def train_models(self):
        '''
        train, store model results: images + scores, and store models
        '''
        # grid search
        self.cl_vars.rfc = RandomForestClassifier(random_state=42)
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference:
        # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        self.cl_vars.lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(
            estimator=self.cl_vars.rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(self.cl_vars.X_train, self.cl_vars.y_train)
        self.cl_vars.rfc = cv_rfc.best_estimator_
        self.cl_vars.lrc.fit(self.cl_vars.X_train, self.cl_vars.y_train)

        self.save_models()

        self.cl_vars.y_train_preds_rf = self.cl_vars.rfc.predict(
            self.cl_vars.X_train)
        self.cl_vars.y_test_preds_rf = self.cl_vars.rfc.predict(
            self.cl_vars.X_test)
        self.cl_vars.y_train_preds_lr = self.cl_vars.lrc.predict(
            self.cl_vars.X_train)
        self.cl_vars.y_test_preds_lr = self.cl_vars.lrc.predict(
            self.cl_vars.X_test)

    def infer_models(self):
        """
        Infer Models
        """
        self.load_models()
        self.cl_vars.y_train_preds_rf = self.cl_vars.rfc.predict(
            self.cl_vars.X_train)
        self.cl_vars.y_test_preds_rf = self.cl_vars.rfc.predict(
            self.cl_vars.X_test)

        self.cl_vars.y_train_preds_lr = self.cl_vars.lrc.predict(
            self.cl_vars.X_train)
        self.cl_vars.y_test_preds_lr = self.cl_vars.lrc.predict(
            self.cl_vars.X_test)


if __name__ == "__main__":
    pass
    churn_lib = ChurnLibrary("/data/MLOps/Customer_Churn/data/bank_data.csv")
    churn_lib.perform_eda()
    churn_lib.encoder_helper_loop()
    churn_lib.perform_feature_engineering()
    churn_lib.train_models()
    # churn_lib.infer_models()
    churn_lib.roc_plot()
    churn_lib.feature_importance_plot()
    churn_lib.classification_report_image()
