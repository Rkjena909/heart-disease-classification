# Dataset

Supplied CSV: 920 rows and 16 columns. File copied unchanged from the project archive.

Source used in the submitted project: [UCI Heart Disease Data, Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data). Original data: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). [Heart Disease, UCI Machine Learning Repository](https://doi.org/10.24432/C52P4X), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The combined supplied file differs from UCI's 303-record Cleveland-only download.

Target: `num = 0` is negative; `num > 0` is positive. Thirteen clinical predictors are used. `id` is excluded; `dataset` is retained for cohort evaluation only. No names or direct patient contact details are present in this CSV.

Numeric predictors: age, trestbps, chol, thalch, oldpeak, ca. Categorical predictors: sex, cp, fbs, restecg, exang, slope, thal.

Main comparison retains zero values. A separate training-only sensitivity analysis treats zero cholesterol and resting blood pressure as missing; this is an analysis assumption, not a verified correction of individual records.
