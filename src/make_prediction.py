# from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# # Step 1: Create input data
# data = CustomData(
#     gender="male",
#     race_ethnicity="group B",
#     parental_level_of_education="bachelor's degree",
#     lunch="standard",
#     test_preparation_course="completed",
#     reading_score=75,
#     writing_score=80
# )

# # Step 2: Convert input into DataFrame
# pred_df = data.get_data_as_data_frame()

# print("Input Data:")
# print(pred_df)

# # Step 3: Initialize prediction pipeline
# predict_pipeline = PredictPipeline()

# # Step 4: Make prediction
# results = predict_pipeline.predict(pred_df)

# print("\nPrediction:")
# print(results[0])

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

gender = input("Enter gender: ")
race_ethnicity = input("Enter race/ethnicity: ")
parental_education = input("Enter parental education: ")
lunch = input("Enter lunch type: ")
test_preparation = input("Enter test preparation course: ")

reading_score = float(input("Enter reading score: "))
writing_score = float(input("Enter writing score: "))

# Create CustomData object
data = CustomData(
    gender=gender,
    race_ethnicity=race_ethnicity,
    parental_level_of_education=parental_education,
    lunch=lunch,
    test_preparation_course=test_preparation,
    reading_score=reading_score,
    writing_score=writing_score
)

# Convert to DataFrame
pred_df = data.get_data_as_data_frame()

# Make prediction
predict_pipeline = PredictPipeline()
results = predict_pipeline.predict(pred_df)

print(results)
print("\nPredicted Math Score:", results[0])