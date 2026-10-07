# Historical Presentation Transcript

Text extracted from the original academic presentation. Student IDs omitted. This historical material contains claims and scores superseded by the revised analysis. Original slide formatting and graphics are not reproduced here.

## Slide 1

HEART DISEASE PREDICTION USING MACHINE LEARNING

Research Paper Presentation by

:

Rahul Kumar Jena

Deeksha

Abbadasari

Meghna

Kattekola

Kondur

Datha

Vaishnavi

Aditya Dev MAMIDI.

## Slide 2

INTRODUCTION

With its transformational potential, machine learning (ML) offers until unheard-of possibilities for redefining cardiovascular health therapies.

primarily seeks to use ML skills to greatly advance the areas

.

Primary

O

bjective -

creation of a strong prediction model using knowledge from the extensive Heart Disease UCI dataset to aid in the early diagnosis of heart disorders

.

## Slide 3

Why this research?

The research mainly focuses on early prediction of Heart diseases through various ML algorithms such as

Logistic Regression, Decision

Trees, Random Forests, Gradient Boosting Machines, Neural Networks, and Support Vector Machines.

## Slide 4

OVERVIEW OF DATASET

The multivariate dataset being examined includes a wide range of statistical and mathematical factors related to cardiovascular health

.

Despite initially consisting of 76 attributes, studies have predominantly focused on a subset of 14 attributes such as

age, sex,

chest pain type, and others.

We aim

to conduct experimental tasks, extract insights to enhance understanding and diagnosis of cardiovascular issues.

Link to the dataset:

https://www.kaggle.com/datasets/redwankarimsony/heartdisease-data?resource=download

## Slide 5

HYPOTHESIS

According to the theory, these traits have observable links and patterns that the machine learning algorithm may pick up on and apply to fresh, unobserved data.

.

If this hypothesis is validated successfully, it will help construct a prediction model that is useful for diagnosing cardiac disease.

.

## Slide 6

PROBLEM DESCRIPTION

The

study delves into the intricacies of developing a robust machine learning-based predictive model for heart disease detection, utilizing insights derived from the Heart Disease UCI

dataset.

The

preprocessing hurdles require meticulous

consideration to ensure data integrity and suitability for

subsequent machine learning algorithms.

A critical aspect of the research involves a comprehensive exploration of various machine learning algorithms, ranging from Logistic Regression to Support Vector Machines.

## Slide 7

RESEARCH AIMS

Create a predictive machine learning model for the early detection of heart disease by utilizing the Heart Disease UCI dataset.

Improve early interventions in cardiovascular health by ensuring timely and precise interventions, without differentiation among various departments within the same organization.

## Slide 8

RESEARCH OBJECTIVES:

Objective 1: Understand and preprocess the Heart Disease UCI dataset for applying machine learning model..

Objective 2: Evaluate the effectiveness of several machine learning techniques, such as Logistic Regression, Decision Trees, Random Forests, Gradient Boosting Machines, Neural Networks, and Support.

## Slide 9

Contd.

Objective 3: Improve accuracy and robustness by using ensemble techniques to aggregate predictions from several models.

Objective 4: Evaluate models based on measures including accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.

Objective 5: Ensure the final model is interpretable, enabling healthcare practitioners to comprehend its predictions.

## Slide 10

RESEARCH QUESTIONS

How effective are various machine learning algorithms in predicting heart disease using the Heart Disease UCI dataset?

The various Machine Learning algorithms predicted decent accuracy when compared to each other, the following are presented below:

Logistic Regression:

Accuracy: 81.52%

F1 Score: 83.96%

Recall: 81.65%

ROC-AUC: 83.79%

## Slide 11

RESEARCH QUESTIONS (

Contd.)

Random Forest:

Accuracy: 85.33%

F1 Score: 87.20%

Recall: 84.40%

ROC-AUC: 91.09%

Support Vector Machine (SVM):

Accuracy: 82.61%

F1 Score: 85.05%

Recall: 83.49%

ROC-AUC: 82.32%

## Slide 12

RESEARCH QUESTIONS (Contd.)

Decision Tree:

Accuracy: 77.72%

F1 Score: 79.60%

Recall: 73.39%

ROC-AUC: 78.70%

Neural Network:

Accuracy: 80.43%

F1 Score: 83.33%

Recall: 82.57%

ROC-AUC: 81.74%

## Slide 13

RESEARCH QUESTIONS (Contd.)

From the above model results and accuracy percentages, we can summarize that:

Best Overall Performance: Random Forest stands out with the highest scores across all metrics.

Balance of Metrics: SVM and logistic regression provide a decent balance of ROC, accuracy, and recall.-AUC.

Potential for Improvement: While somewhat efficient, decision trees and neural networks indicate regions where classification accuracy and ROC-AUC scores might be improved..

## Slide 14

RQ2: What preprocessing techniques, including handling missing values, encoding categorical variables, and normalizing numerical attributes, are most effective for this dataset?

Hyperparameter tuning is the process of optimizing the hyperparameters of a machine learning model to achieve better performance. Hyperparameters are configuration settings external to the model that cannot be learned from the data but significantly impact the model's performance.

## Slide 15

1. Accuracy: 85.87%

shows the percentage of all forecasts that were accurate (heart disease and no heart disease combined). High accuracy indicates that the model can accurately determine if cardiac disease is present or not.

2. F1 Score: 87.96%

a balance between recall and accuracy that shows how well the model can detect genuine positive situations while reducing false positives. It is noteworthy when an F1 score approaches 88%, particularly in the field of medical diagnostics where false positives and negatives may have serious consequences..

3. Recall: 87.16%

demonstrates the model's capacity to detect actual positive heart disease cases.  In medical contexts, this high recall rate is essential to guarantee that the majority of individuals with the illness are appropriately recognized.

## Slide 16

4. ROC-AUC: 91.18%

Measures the model's ability to distinguish between classes (heart disease vs. no heart disease) across different thresholds. An AUC of over 91% is excellent, indicating a high degree of separability achieved by the model.

ROC Curve Analysis

The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate for different threshold values.

The area under the curve (AUC) being approximately 91% reflects the model's high discriminative ability.

The curve staying well above the diagonal line (representing a random guess) visually confirms the model's effectiveness.

## Slide 17

RQ3: How does the implementation of ensemble techniques impact the accuracy and robustness of heart disease predictions?

Comparing the performance metrics of the optimized Random Forest model with the initial model yields. Accuracy

## Slide 18

Optimized Model: 85.87%

Initial Model: 85.33%

Analysis: There's a slight improvement in accuracy with the optimized model. This indicates a marginally better overall rate of correct predictions for both classes.

F1 Score

Optimized Model: 87.96%

Initial Model: 87.20%

Analysis: The F1 score, which balances precision and recall, is higher in the optimized model. This suggests a better balance in correctly identifying positive cases while minimizing false positives.

## Slide 19

Recall

Optimized Model: 87.16%

Initial Model: 84.40%

Analysis: There's a notable improvement in recall with the optimized model. This is significant in medical diagnostics, as a higher recall means the model is more effective at identifying true positive cases of heart disease.

ROC-AUC

Optimized Model: 91.18%

Initial Model: 91.09%

Analysis: Both models exhibit an excellent ROC-AUC score, with a marginal improvement in the optimized model. This score reflects the model's ability to distinguish between the classes at various thresholds, and a higher score indicates better model performance.

## Slide 20

IMPACTS OF STUDY IN THE MEDICAL FIELD

A) Early Detection and Intervention: Early identification of cardiac problems is made possible by the creation of a reliable machine learning-based prediction model. Proactive actions may be facilitated by the model's timely alarms, which can lower the likelihood of serious cardiac events and eventually improve patient outcomes.

B) Algorithmic Evaluation: The thorough investigation of several machine learning algorithms, such as Support Vector Machines, Gradient Boosting Machines, Random Forests, Decision Trees, and Logistic Regression, offers insightful information about how well they work in the complex field of heart disease prediction. This assessment helps choose the best models for precise forecasts.

## Slide 21

C) Personalized Medicine: With the use of machine learning, it is possible to analyze patient data individually and customize therapies according to lifestyle, genetic composition, and other unique characteristics. This method improves overall patient outcomes, minimizes adverse effects, and increases therapeutic efficacy.

## Slide 22

CONCLUSION

Although the gains are slight, the Random Forest model's hyperparameter adjustment has improved all important performance indicators. This implies that the original model was already operating rather well, but that it has marginally improved in terms of its accuracy in predicting heart disease via tweaking. In a medical setting, the higher recall rate is especially significant since it suggests a greater chance of accurately identifying people who have heart problems.

## Slide 23

REFERENCES

Abdalrada

, A. S.,

Abawajy

, J., Al-Quraishi, T., & Islam, S. M. S. (2022). Machine learning models for prediction of co-occurrence of diabetes and cardiovascular diseases: a retrospective cohort study. International Journal of Medical Informatics, 23(1), 112-120. Retrieved on September 6, 2023, from https://pubmed.ncbi.nlm.nih.gov/35673486

World Health Organization. (2021). Cardiovascular diseases (CVDs). Retrieved on September 7, 2023, from https://www.who.int/news-room/fact- sheets/detail/cardiovascular-diseases-(

cvds

)

Qian, X., Li, Y., Zhang, X., Guo, H., He, J., Wang, X., Yan, Y., Ma, J., Ma, R., & Guo, S. (2022). A Cardiovascular Disease Prediction Model Based on Routine Physical Examination Indicators Using Machine Learning Methods: A Cohort Study. Retrieved on September 8, 2023, from https://pubmed.ncbi.nlm.nih.gov/35783868

Jiang, L., Chen, S., Wu, Y., Zhou, D., & Duan, L. (2021). Prediction of coronary heart disease in gout patients using machine learning models. Mathematical Biosciences and Engineering. https://doi.org/10.3934/mbe.2023212. Accessed on 6th September 2023.

## Slide 24

THANK YOU