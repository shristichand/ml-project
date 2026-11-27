from src.pipeline.predict_pipeline import (
    BackgroundPredictPipeline,
    CustomData,
    PredictPipeline,
)

gender = input("Enter gender: ")
race_ethnicity = input("Enter race/ethnicity: ")
parental_education = input("Enter parental education: ")
lunch = input("Enter lunch type: ")
test_preparation = input("Enter test preparation course: ")

reading_score = float(input("Enter reading score: "))
writing_score = float(input("Enter writing score: "))

data = CustomData(
    gender=gender,
    race_ethnicity=race_ethnicity,
    parental_level_of_education=parental_education,
    lunch=lunch,
    test_preparation_course=test_preparation,
    reading_score=reading_score,
    writing_score=writing_score,
)

pred_df = data.get_data_as_data_frame()
math_details = PredictPipeline().predict_math_details(pred_df)
print("\nPredicted Math Score:", math_details["predicted_math_score"])
print("Math risk band:", math_details["risk"]["label"])

background = BackgroundPredictPipeline().predict_details(data.get_background_frame())
print("Predicted average (background only):", background["predicted_average_score"])
print("Overall risk band:", background["risk"]["label"])
