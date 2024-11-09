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
