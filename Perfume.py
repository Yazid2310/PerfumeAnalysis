import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from geopandas.tools import geocode
from geopy.geocoders import Nominatim
import folium
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor



# Import data
men_perfume = pd.read_csv('ebay_mens_perfume.csv')
women_perfume = pd.read_csv('ebay_womens_perfume.csv')

# Reorganize and clean the data
men_perfume['gender'] = 'men'
women_perfume['gender'] = 'women'

all_perfume = pd.concat([men_perfume, women_perfume], ignore_index=True)

print(all_perfume.columns)
print(all_perfume.head())

all_perfume = all_perfume.fillna({
    'brand': 'Unknown',
    'type': 'Unknown',
    'available': 0,
    'availableText': 'Not available',
    'sold': 0,
    'lastUpdated': 'Unknown'})

print((all_perfume.isnull().sum()))
print(all_perfume.dtypes)

# Plotting the price range histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=all_perfume, x='price', kde=True, bins=30, color='skyblue')

plt.title('Perfume Prices Distribution')
plt.xlabel('Price')
plt.ylabel('Number of products listed')

plt.show()

# Calculate the average price, highest price, and lowest price
average_price = all_perfume['price'].mean()
highest_price = all_perfume['price'].max()
lowest_price = all_perfume['price'].min()

# Print the results
print(f"The average price is: ${average_price:.2f}")
print(f"The highest price is: ${highest_price:.2f}")
print(f"The lowest price is: ${lowest_price:.2f}")

# Gender analysis
gender_counts = all_perfume['gender'].value_counts()
print(f'Number of each {gender_counts} : ')

# Calculate the average price for men and women perfumes
average_price_men = all_perfume[all_perfume['gender'] == 'men']['price'].mean()
average_price_women = all_perfume[all_perfume['gender'] == 'women']['price'].mean()

# Print the average prices
print(f"The average price for men's perfume is: ${average_price_men:.2f}")
print(f"The average price for women's perfume is: ${average_price_women:.2f}")

# Visualize the average prices
plt.figure(figsize=(8, 5))
fig, ax = plt.subplots()
colors = sns.color_palette('viridis', 2)
ax.bar(['Men', 'Women'], [average_price_men, average_price_women], color=colors)

plt.title('Average Perfume Price by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Price')
plt.show()

# Calculate the total sales for men and women perfumes
total_sales_men = all_perfume[all_perfume['gender'] == 'men']['sold'].sum()
total_sales_women = all_perfume[all_perfume['gender'] == 'women']['sold'].sum()

# Print the total sales
print(f"The total sales for men's perfume is: {total_sales_men}")
print(f"The total sales for women's perfume is: {total_sales_women}")

# Visualize the total sales
plt.figure(figsize=(8, 5))
fig, ax = plt.subplots()
ax.bar(['Men', 'Women'], [total_sales_men, total_sales_women], color=colors)

plt.title('Total Perfume Sales by Gender')
plt.xlabel('Gender')
plt.ylabel('Total Sales')
plt.show()

# Calculate the total sales
total_sales = total_sales_men + total_sales_women

# Calculate the percentage sales
percent_sales_men = (total_sales_men / total_sales) * 100
percent_sales_women = (total_sales_women / total_sales) * 100

# Print the percentage sales
print(f"Percentage of sales for men's perfume: {percent_sales_men:.2f}%")
print(f"Percentage of sales for women's perfume: {percent_sales_women:.2f}%")

# Visualize the percentage sales in a pie chart
plt.figure(figsize=(8, 5))
labels = ['Men', 'Women']
sales_percentages = [percent_sales_men, percent_sales_women]
colors = sns.color_palette('viridis', len(labels))
plt.pie(sales_percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Total Perfume Sales by Gender (Percentage)')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

# Calculate the total sales and average price for each brand
brand_sales = all_perfume.groupby('brand')['sold'].sum()
brand_avg_price = all_perfume.groupby('brand')['price'].mean()

# Print total sales for each brand
print("Total sales for each brand:")
for brand, sales in brand_sales.items():
    print(f"{brand}: {sales}")

# Print average price for each brand
print("\nAverage price for each brand:")
for brand, avg_price in brand_avg_price.items():
    print(f"{brand}: ${avg_price:.2f}")

# Identify the top 10 brands by sales
top_brands = brand_sales.nlargest(10)

# Print the top 10 brands by sales
print("\nTop 10 brands by sales:")
for brand, sales in top_brands.items():
    print(f"{brand}: {sales}")

# Filter data for the top 10 brands
top_brand_avg_price = brand_avg_price[top_brands.index]
top_brand_sales = brand_sales[top_brands.index]

# Visualize the relationship between the top 10 brand popularity and pricing with a bubble chart
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=top_brand_avg_price,
    y=top_brand_sales,
    size=top_brand_sales,
    sizes=(20, 2000),
    legend=False,
    alpha=0.6
)

# Annotate each point with the brand name
for brand in top_brand_avg_price.index:
    plt.text(
        x=top_brand_avg_price[brand],
        y=top_brand_sales[brand],
        s=brand,
        fontsize=9,
        ha='right',
        va='bottom'
    )

plt.title('Relationship Between Top 10 Brand Popularity and Pricing')
plt.xlabel('Average Price')
plt.ylabel('Total Sales')
plt.show()

# Calculate the total sales for each perfume type
type_sales = all_perfume.groupby('type')['sold'].sum()

# Identify the top 10 types by sales
top_types = type_sales.nlargest(10)

# Print the top 10 types by sales
print("\nTop perfume types by sales:")
for perfume_type, sales in top_types.items():
    print(f"{perfume_type}: {sales}")

# Visualize the top 10 perfume types by sales
plt.figure(figsize=(12, 8))
sns.barplot(x=top_types.values, y=top_types.index, hue=top_types.index, palette="viridis", legend=False)

plt.title('Top  10 Perfume Types by Sales')
plt.xlabel('Total Sales')
plt.ylabel('Perfume Type')



# Calculate the correlation between availability and sales
availability_sales_corr = all_perfume['available'].corr(all_perfume['sold'])
print(f"Correlation between availability and sales: {availability_sales_corr:.2f}")

# Visualize the correlation using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=all_perfume, x='available', y='sold', alpha=0.6)

plt.title('Correlation Between Availability and Sales')
plt.xlabel('Availability')
plt.ylabel('Total Sales')

# Choose features and target variable
features = ['price', 'available', 'type', 'brand', 'gender']
X = all_perfume[features]
y = all_perfume['sold']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up preprocessing pipelines
numeric_features = ['price', 'available']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_features = ['type', 'brand', 'gender']
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Feature selection: Select the top k features
k = 100
feature_selector = SelectKBest(f_regression, k=k)

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', feature_selector),
    ('regressor', GradientBoostingRegressor(n_estimators=450, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Extract and group feature importances
importance = model.named_steps['regressor'].feature_importances_

# Map original features back to their summed importance scores
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
selected_features = feature_names[feature_selector.get_support()]

# Summing importance scores for original features
summed_importances = {
    'price': sum(importance[i] for i, f in enumerate(selected_features) if 'price' in f),
    'available': sum(importance[i] for i, f in enumerate(selected_features) if 'available' in f),
    'type': sum(importance[i] for i, f in enumerate(selected_features) if 'type' in f),
    'brand': sum(importance[i] for i, f in enumerate(selected_features) if 'brand' in f),
    'gender': sum(importance[i] for i, f in enumerate(selected_features) if 'gender' in f)
}

# Plot the summarized feature importances
plt.bar(summed_importances.keys(), summed_importances.values())
plt.xlabel('Features')
plt.ylabel('Summed Importance Score')
plt.title('Distribution of Summed Feature Importance')
plt.show()

# Count the occurrences of each location
location_counts = all_perfume['itemLocation'].value_counts()
# Print the top locations by count
print("\nItem locations by count:")
for location, count in location_counts.items():
    print(f"{location}: {count}")


# Initialize geolocator and dictionary for coordinates#
geolocator = Nominatim(user_agent="perfume_map")
location_coords = {}

# Convert location names to coordinates, if not already known
for location in location_counts.index:
    try:
        loc = geolocator.geocode(location)
        if loc:
            location_coords[location] = (loc.latitude, loc.longitude)
        time.sleep(1)  # To avoid hitting request limits
    except Exception as e:
        print(f"Geocoding error for {location}: {e}")

# Initialize map centered around a mean global location
m = folium.Map(location=[20, 0], zoom_start=2)

# Add markers for each location based on the sales count
for location, count in location_counts.items():
    if location in location_coords:
        lat, lon = location_coords[location]
        folium.CircleMarker(
            location=(lat, lon),
            radius=min(count * 0.1, 10),  # Scale radius by count
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
            popup=f"{location}: {count} sales",
        ).add_to(m)

# Save the map as an HTML file
m.save("perfume_sales_map.html")



# Define target (y) and features (X)
y = all_perfume['sold']
X = all_perfume.drop(columns=['sold', 'lastUpdated', 'availableText'])  # Drop non-relevant columns

# Define categorical and numerical features
categorical_features = ['brand', 'type', 'gender', 'itemLocation']
numeric_features = ['price', 'available']

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine transformations
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model pipeline with preprocessing and the regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=0))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Extract feature importances
importance = model.named_steps['regressor'].feature_importances_

# Get feature names
feature_names = numeric_features + list(model.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_features))

# Create a DataFrame to organize feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

