# Final Insights

## EDA Observations
- `University_%`,`Junior_col_%`,`Salary_Log`are the most significant factor, with a strong positive correlation (>0.4) to life expectancy.
- Salary shows a complex, nonlinear relationship with life expectancy (further investigation required).
- Using VIF theres a severe multicolinearity between  `University_%`,`Junior_col_%`,`Salary_Log`

## Feature Engineering
- Applied log transformation to `Salary` and `park_land_%`to reduce skewness

## Model Evaluation
- The Log_transformed `Salary` and `Park_land_%` Model provides a more accurate prediction of `Life_expectancy` for test_data (Higher R2 and lower RMSE) than the other models
- This confirms the non linearity between `Salary` and `Life_expectancy`
- model_v2 provides the best prediction result (R2 and RMSE)

## Feature importance and Visualization
- SHAP analysis confirms `University_%` and `Salary` as key predictors for `Life_expectancy`
- `University_%` and `Salary` have the most significant impact on predicting `Life_expectancy` 
- Higher Values for `University_%`,`Physician_100Kp`,`Park_Land_%`Increases `Life_expectancy` while lower values Decrease it
- Higher education consistently improves life expectancy while salary has a mixed effect.

## Recommendations
- Increase access to higher education to improve life expectancy.
- Investigate urbanization factors that may negatively impact life expectancy despite high salaries (Stress, poor diet, lack of activity).
- 
