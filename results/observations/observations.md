# Final Insights

## EDA Observations
- `University_%`,`Junior_col_%`, are the most significant factors, with a strong positive correlation (>0.4) to life expectancy.
- Salary shows a complex, nonlinear relationship with life expectancy (further investigation required).
- Using VIF theres a severe multicolinearity between  `University_%`,`Junior_col_%`,`Salary_Log`

## Feature Engineering
- Applied log transformation to `Salary` and `park_land_%`to reduce skewness

## Model Evaluation
- The Log_transformed `Salary` and `Park_land_%` Model provides a more accurate prediction of `Life_expectancy` for test_data (Higher R2 and lower RMSE) than the other models
- This confirms the non linearity between `Salary` and `Life_expectancy`
- model trained with `University_%`, `Junior_col_%` provides the best prediction result (R2 and RMSE)

## Feature importance and Visualization
- SHAP analysis confirms `University_%` and `Junior_col_%` as key predictors for `Life_expectancy`
- Higher Values for `University_%`,`Junior_col_%%`Increases `Life_expectancy` while lower values Decrease it
- Higher education consistently improves life expectancy while salary has a mixed effect.

## Recommendations
- Linear regression Model trained on `University_%`, `Junior_col_%` provides the most accurate `Life_expectancy`prediction therefore  Increasing access to higher education is linked to improving life expectancy (Correlation not causation)
- Data reveals that a higher education consistently improves life expectancy while lower education has a negative impact on life expectancy. 
- Investigate urbanization factors that may negatively impact life expectancy perhaps due to(Stress, poor diet, lack of activity)
  