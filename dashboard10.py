import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import plotly.express as px

# Initialize SparkSession
spark = SparkSession.builder.appName("InteractiveDashboard").getOrCreate()

# Read the data
data = spark.read.format("csv").option("header", "true").load("joined_df3.csv")

# Get the column names
column_names = data.columns

# Set the column names
data = data.toDF(*column_names)

# Convert numeric columns to appropriate data types
numeric_columns = ['YearlyIncome(USD)', 'TotalChildren', 'NumberChildrenAtHome', 'NumberCarsOwned',
                   'AnnualSales(USD)', 'MinPaymentAmount(USD)', 'AnnualRevenue(USD)', 'HouseOwnerFlag']
for column in numeric_columns:
    data = data.withColumn(column, data[column].cast('double'))

# Create filters for Marital Status and Gender
marital_status_filter = st.sidebar.selectbox("Select Marital Status", ['All'] + [row.MaritalStatus for row in data.select("MaritalStatus").distinct().collect()])
gender_filter = st.sidebar.selectbox("Select Gender", ['All'] + [row.Gender for row in data.select("Gender").distinct().collect()])

# Apply filters to the data
filtered_data = data
if marital_status_filter != 'All':
    filtered_data = filtered_data.filter(filtered_data['MaritalStatus'] == marital_status_filter)
if gender_filter != 'All':
    filtered_data = filtered_data.filter(filtered_data['Gender'] == gender_filter)

# Show filtered data as a table
st.dataframe(filtered_data.toPandas().rename(columns=dict(zip(filtered_data.columns, column_names))))

# Create a histogram for Yearly Income
st.subheader("Yearly Income Distribution")
income_data = [row["YearlyIncome(USD)"] for row in filtered_data.select("YearlyIncome(USD)").collect()]
st.plotly_chart(px.histogram(x=income_data, nbins=20))

# Create a bar chart for Marital Status
st.subheader("Marital Status")
marital_status_counts = filtered_data.groupBy('MaritalStatus').count().collect()
marital_status_counts_df = {
    'MaritalStatus': [row.MaritalStatus for row in marital_status_counts],
    'count': [row['count'] for row in marital_status_counts]
}
st.plotly_chart(px.bar(marital_status_counts_df, x='MaritalStatus', y='count'))

# Create a pie chart for Gender
st.subheader("Gender")
gender_counts = filtered_data.groupBy('Gender').count().collect()
gender_counts_df = {
    'Gender': [row.Gender for row in gender_counts],
    'count': [row['count'] for row in gender_counts]
}
st.plotly_chart(px.pie(gender_counts_df, values='count', names='Gender'))

# Create scatter plot for Yearly Income vs. Annual Sales
st.subheader("Yearly Income vs. Annual Sales")
scatter_data = filtered_data.select('YearlyIncome(USD)', 'AnnualSales(USD)').collect()
scatter_data_df = {
    'YearlyIncome(USD)': [row['YearlyIncome(USD)'] for row in scatter_data],
    'AnnualSales(USD)': [row['AnnualSales(USD)'] for row in scatter_data]
}
st.plotly_chart(px.scatter(scatter_data_df, x='YearlyIncome(USD)', y='AnnualSales(USD)', title='Yearly Income vs. Annual Sales',
                           labels={'YearlyIncome(USD)': 'Yearly Income In USD', 'AnnualSales(USD)': 'Annual Sales In USD'}))

# Average Yearly Income by Education Level
avg_income_by_education = data.groupBy('EnglishEducation').avg('YearlyIncome(USD)').collect()
avg_income_by_education_df = {
    'EnglishEducation': [row.EnglishEducation for row in avg_income_by_education],
    'avg(YearlyIncome(USD))': [row['avg(YearlyIncome(USD))'] for row in avg_income_by_education]
}
st.plotly_chart(px.bar(avg_income_by_education_df, x='EnglishEducation', y='avg(YearlyIncome(USD))',
                       title='Average Yearly Income by Education Level In USD'))

# Perform Linear Regression for Annual Sales
vector_assembler = VectorAssembler(inputCols=['YearlyIncome(USD)', 'TotalChildren', 'NumberChildrenAtHome',
                                              'NumberCarsOwned', 'MinPaymentAmount(USD)', 'AnnualRevenue(USD)',
                                              'HouseOwnerFlag'], outputCol='features')
assembled_data = vector_assembler.transform(filtered_data)
lr = LinearRegression(featuresCol='features', labelCol='AnnualSales(USD)', regParam=0.01)
model = lr.fit(assembled_data)
predictions = model.transform(assembled_data)

# Perform Linear Regression for Annual Sales
vector_assembler = VectorAssembler(inputCols=['YearlyIncome(USD)', 'TotalChildren', 'NumberChildrenAtHome',
                                              'NumberCarsOwned', 'MinPaymentAmount(USD)', 'AnnualRevenue(USD)',
                                              'HouseOwnerFlag'], outputCol='features')
assembled_data = vector_assembler.transform(filtered_data)
lr = LinearRegression(featuresCol='features', labelCol='AnnualSales(USD)', regParam=0.01)
model = lr.fit(assembled_data)
predictions = model.transform(assembled_data)

# Create line plot for Predicted Sales
st.subheader("Predicted Sales")
prediction_data = predictions.select('AnnualSales(USD)', 'prediction').collect()
prediction_data_df = {
    'AnnualSales(USD)': [row['AnnualSales(USD)'] for row in prediction_data],
    'prediction': [row['prediction'] for row in prediction_data]
}
st.plotly_chart(px.line(prediction_data_df, x='AnnualSales(USD)', y='prediction', title='Predicted Sales',
                        labels={'AnnualSales(USD)': 'Annual Sales In USD', 'prediction': 'Predicted Sales'}))

# Scatter Plot of Predicted vs Actual Annual Sales
st.subheader("Predicted vs Actual Annual Sales")
st.plotly_chart(px.scatter(prediction_data_df, x='AnnualSales(USD)', y='prediction',
                           title='Predicted vs Actual Annual Sales In USD'))