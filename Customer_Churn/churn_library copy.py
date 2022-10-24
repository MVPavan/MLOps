# # library doc string


# # import libraries
# import shap
# import joblib
# import pytest
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()

# from sklearn.preprocessing import normalize
# from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import plot_roc_curve, classification_report
# import os
# from pathlib import Path
# import constansts as consts
# os.environ['QT_QPA_PLATFORM']='offscreen'

# images_path = Path(__file__).parent/"images"
# images_path.mkdir(parents=True, exist_ok=True)

# class ChurnLibrary:
#     def __init__(self) -> None:
#         pass
#     def import_data(self, pth):
#         '''
#         returns dataframe for the csv found at pth
#         input:
#                 pth: a path to the csv
#         output:
#                 df: pandas dataframe
#         '''	
#         return pd.read_csv(pth)

# def plot_func(df_column, plot_type, save_name):
#     """
#     df_column: series to plot
#     plot_type: ["hist","bar","sns_hist_density"]
#     return: matplotlib figure
#     """
#     plt.figure(figsize=(20,10)) 
#     if plot_type=="hist":
#         fig = df_column.hist()
#     if plot_type=="bar":
#         fig = df_column.plot(kind="bar")
#     if plot_type=="sns_hist_density":
#         # fig = sns.histplot(df_column, stat='density', kde=True)
#         fig = sns.distplot(df_column, kde=True)
#     if plot_type=="sns_heatmap":
#         fig = sns.heatmap(df_column.corr(), annot=False, cmap='Dark2_r', linewidths = 2)

#     fig.get_figure().savefig(images_path/save_name)
#     return
        

# def perform_eda(df, plots=True):
#     '''
#     perform eda on df and save figures to images folder
#     input:
#             df: pandas dataframe

#     output:
#             None
#     '''
#     df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
#     if not plots:return
#     plot_func(
#         df_column=df["Churn"], 
#         plot_type="hist", save_name=consts.eda_plots["Churn"])
#     plot_func(
#         df_column=df["Customer_Age"], 
#         plot_type="hist", save_name=consts.eda_plots["Customer_Age"])
#     plot_func(
#         df_column=df["Marital_Status"].value_counts('normalize'), 
#         plot_type="bar", save_name=consts.eda_plots["Marital_Status"])
#     plot_func(
#         df_column=df['Total_Trans_Ct'], 
#         plot_type="sns_hist_density", save_name=consts.eda_plots["Total_Trans_Ct"])
#     plot_func(
#         df_column=df, 
#         plot_type="sns_heatmap", save_name=consts.eda_plots["heatmap"])
#     return


# def encoder_helper_loop(df, category_lst=consts.cat_columns, response=None):
#     '''
#     helper function to turn each categorical column into a new column with
#     propotion of churn for each category - associated with cell 15 from the notebook

#     input:
#             df: pandas dataframe
#             category_lst: list of columns that contain categorical features
#             response: string of response name [optional argument that could be used for naming variables or index y column]

#     output:
#             df: pandas dataframe with new columns for
#     '''
#     for cat_col in category_lst:
#         df[f"{cat_col}_Churn"] = df[cat_col].map(df.groupby(cat_col).mean(numeric_only=True)['Churn'])
#         consts.quant_columns.append(f"{cat_col}_Churn")
#     return df


# def perform_feature_engineering(df, response=None):
#     '''
#     input:
#               df: pandas dataframe
#               response: string of response name [optional argument that could be used for naming variables or index y column]

#     output:
#               X_train: X training data
#               X_test: X testing data
#               y_train: y training data
#               y_test: y testing data
#     '''
#     X = df[consts.quant_columns] 
#     y = df["Churn"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
#     return X_train, X_test, y_train, y_test


# def classification_report_image(y_train,
#                                 y_test,
#                                 y_train_preds_lr,
#                                 y_train_preds_rf,
#                                 y_test_preds_lr,
#                                 y_test_preds_rf):
#     '''
#     produces classification report for training and testing results and stores report as image
#     in images folder
#     input:
#             y_train: training response values
#             y_test:  test response values
#             y_train_preds_lr: training predictions from logistic regression
#             y_train_preds_rf: training predictions from random forest
#             y_test_preds_lr: test predictions from logistic regression
#             y_test_preds_rf: test predictions from random forest

#     output:
#              None
#     '''
#     plt.rc('figure', figsize=(5, 5))
#     #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
#     plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
#     plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
#     plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
#     plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
#     plt.axis('off')

#     plt.rc('figure', figsize=(5, 5))
#     plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
#     plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
#     plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
#     plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
#     plt.axis('off')


# def feature_importance_plot(model, X_data, output_pth):
#     '''
#     creates and stores the feature importances in pth
#     input:
#             model: model object containing feature_importances_
#             X_data: pandas dataframe of X values
#             output_pth: path to store the figure

#     output:
#              None
#     '''
#     explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
#     shap_values = explainer.shap_values(X_test)
#     shap.summary_plot(shap_values, X_test, plot_type="bar")

#     # Calculate feature importances
#     importances = cv_rfc.best_estimator_.feature_importances_
#     # Sort feature importances in descending order
#     indices = np.argsort(importances)[::-1]

#     # Rearrange feature names so they match the sorted feature importances
#     names = [X.columns[i] for i in indices]

#     # Create plot
#     plt.figure(figsize=(20,5))

#     # Create plot title
#     plt.title("Feature Importance")
#     plt.ylabel('Importance')

#     # Add bars
#     plt.bar(range(X.shape[1]), importances[indices])

#     # Add feature names as x-axis labels
#     plt.xticks(range(X.shape[1]), names, rotation=90);


# def roc_plot(model, X_data, output_pth):
#     '''
#     creates and stores the feature importances in pth
#     input:
#             model: model object containing feature_importances_
#             X_data: pandas dataframe of X values
#             output_pth: path to store the figure

#     output:
#              None
#     '''
#     lrc_plot = plot_roc_curve(lrc, X_test, y_test)

#     plt.figure(figsize=(15, 8))
#     ax = plt.gca()
#     rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
#     lrc_plot.plot(ax=ax, alpha=0.8)
#     plt.show()

# def train_models(X_train, X_test, y_train, y_test):
#     '''
#     train, store model results: images + scores, and store models
#     input:
#               X_train: X training data
#               X_test: X testing data
#               y_train: y training data
#               y_test: y testing data
#     output:
#               None
#     '''
#     # grid search
#     rfc = RandomForestClassifier(random_state=42)
#     # Use a different solver if the default 'lbfgs' fails to converge
#     # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#     lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

#     param_grid = { 
#         'n_estimators': [200, 500],
#         'max_features': ['auto', 'sqrt'],
#         'max_depth' : [4,5,100],
#         'criterion' :['gini', 'entropy']
#     }

#     cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
#     cv_rfc.fit(X_train, y_train)

#     lrc.fit(X_train, y_train)

#     y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
#     y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

#     y_train_preds_lr = lrc.predict(X_train)
#     y_test_preds_lr = lrc.predict(X_test)



# if __name__ == "__main__":
#     # df = import_data("/data/MLOps/Customer_Churn/data/bank_data.csv")
#     # perform_eda(df, plots=False)
#     # df = encoder_helper_loop(df)
#     # X_train, X_test, y_train, y_test = 
#     pass
