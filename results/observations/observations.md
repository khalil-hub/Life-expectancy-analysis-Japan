# Final Insights

## EDA Observations
- Aomori has the worst life expectancy with 82.8
- Shiga has the best life expectancy with 85.5
- Strong positive correlation between `University_%`,`Junior_col_%` and `Life_expectancy` (+0.57)
- Moderately Strong positive correlation between `Physician_100kP`,`Park_land_%`,`Salary` and `Life_expectancy`(~0.3~0.4)
- Strong Negative correlation for `Elementary_school_%` (-0.56),Moderate negative correlation for `Ambulances` suggesting that higher elementary education percentages and higher general hospital beds correspond to lower life expectancy.
- Using VIF theres a severe multicolinearity between  `University_%`,`Salary`

## Feature Engineering
- Applied normalization on all numeric values to reduce skewness
- Using PCA, I combined Salary and University into 1 column called `Socioeconomic_index` to avoid multicolinearity

## Model Evaluation
- The Linear regression Model trained on `Physician_100kP` ,`Junior_college_%` ,`Socioeconomic_index`, `Ambulances_100kP`,`Park_Land_%` showed the best values for R^2=0.37 and RMSE=0.34

## Feature importance and Visualization
- SHAP analysis confirms `Physician_100kP` ,`Junior_college_%` as key predictors for `Life_expectancy`
- Higher Values for `Physician_100kP` ,`Junior_college_%` Increases `Life_expectancy` while lower values Decrease it
- `Park_Land_%`, `Socioeconomic_index` ,`Ambulances_100kP` have a moderate impact on `Life_expectancy`

## Model_experiments:
- Using OLS regression, all three independent variables (`education`, `healthcare access`, and `park access`) are statistically significant predictors of `life expectancy`, with positive relationships.
- `Socioeconomic_index` and `Ambulances_100kP` do not appear to be statistically significant (>0.05) in predicting `life expectancy` based on their p-values.
### Education
- For every 1% increase in the percentage of people with junior college education, life expectancy increases by approximately 1.2 years, assuming other factors remain constant.
- Education plays a crucial role in improving life expectancy. Higher education levels are associated with better access to healthcare, healthier lifestyle choices, and increased awareness of health-related issues, which can contribute to longer life expectancy.
### Healthcare Access
- For every additional physician per 100,000 people, life expectancy increases by 0.70 years, assuming other factors remain constant.
- Access to healthcare is another key determinant of life expectancy. More healthcare providers (physicians) per capita are associated with better health outcomes, as individuals are more likely to receive timely medical care, which can improve overall health and longevity.
### Access to Parks
- For every 1% increase in the percentage of land designated as parks, life expectancy increases by 0.56 years, assuming other factors remain constant
- Access to parks and green spaces contributes to mental and physical well-being, encouraging outdoor physical activities like walking, running, and social interactions, which can reduce stress and improve overall health.

## Recommendations
- Given the strong positive relationship between education and life expectancy, policymakers should prioritize education initiatives. This could involve increasing access to quality education, especially in underserved areas, and promoting higher education to improve overall health awareness.
- The model shows that healthcare access, as measured by the number of physicians per capita, has a significant positive effect on life expectancy. Policymakers should ensure that healthcare services are adequately distributed, especially in rural or underserved areas.
- Access to parks and green spaces significantly influences life expectancy. Urban planning and development should include a focus on creating more parks, recreational areas, and green spaces where residents can engage in physical activities and enjoy nature.
  