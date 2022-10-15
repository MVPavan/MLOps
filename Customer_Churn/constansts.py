# coding=utf-8

cat_columns = {
    'Gender':'Gender_Churn',
    'Education_Level':'Education_Level_Churn',
    'Marital_Status':'Marital_Status_Churn',
    'Income_Category':'Income_Category_Churn',
    'Card_Category':'Card_Category_Churn'
}

quant_columns = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

eda_plots = {
    "Churn":"Churn_Histogram.png",
    "Customer_Age":"Customer_Age_Histogram.png",
    "Marital_Status":"Marital_Status_Bar.png",
    "Total_Trans_Ct":"Total_Trans_Ct.png",
    "heatmap":"Heatmap.png"
}

result_plots = {
    "roc":"RFC_LR_ROC.png",
    "rfc_accuracy":"RFC_Accuracy.png",
    "lr_accuracy":"LR_Accuracy.png",
    "features":"Feature_Importance.png",
    "shap":"Shap_Summary.png"
}

model_names = {
    "rfc":"rfc_model.pkl",
    "lr":"logistic_model.pkl"
}

model_names_pretrained = {
    "rfc":"rfc_model_org.pkl",
    "lr":"logistic_model_org.pkl"
}