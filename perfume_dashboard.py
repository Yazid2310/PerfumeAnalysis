import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from streamlit import navigation
import streamlit.components.v1 as components


# Initialize Streamlit app
def main():
    st.title("Perfume Dashboard")
    # Rest of your app code

if __name__ == "__main__":
    main()

# Load data
men_perfume = pd.read_csv('ebay_mens_perfume.csv')
women_perfume = pd.read_csv('ebay_womens_perfume.csv')
men_perfume['gender'] = 'men'
women_perfume['gender'] = 'women'
all_perfume = pd.concat([men_perfume, women_perfume], ignore_index=True)
all_perfume = all_perfume.fillna({
    'brand': 'Unknown',
    'type': 'Unknown',
    'available': 0,
    'sold': 0
})





st.title("Perfume Sales Dashboard")
st.write('This page presents the results of an analysis '
         'of an e-commerce perfume dataset containing detailed information '
         'on 2000 perfume listings from eBay. The scope of the project is to determine which features '
         'are essential for the customer to sell the most perfumes. For this, I will start with a global '
         'analysis of the data and I will end with a sales prediction model based on the different features, '
         'which are price, availability, type, gender and brand.')




st.sidebar.title('Perfume project analysis')
with st.sidebar:
    st.subheader('Disclaimer')
    st.write('The analysis and insights presented in this portfolio are based '
             'solely on a dataset of e-commerce perfume listings from eBay. The findings '
             'are for academic or work-related purposes and should not be considered as '
             'actionable business insights or generalizable to the broader market. The data '
             'is limited to this specific dataset and is not intended to reflect real-world '
             'trends or consumer behavior outside of this project.')
    st.subheader('Data Source')
    kaggle_link = "https://www.kaggle.com/datasets/kanchana1990/perfume-e-commerce-dataset-2024/data"
    st.markdown(f'<a href="{kaggle_link}" target="_blank" style="color:blue; font-size:16px;">Perfume E-Commerce Dataset 2024 </a>',
        unsafe_allow_html=True)
    st.subheader('Social Networks')
    st.sidebar.markdown("[![Linkedin](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/yazid-aboudou)")
    st.sidebar.markdown(
        """
        <a href="https://github.com/Yazid2310">
            <img src="https://img.icons8.com/material-outlined/48/000000/github.png" width="60" height="60">
        </a>
        """,
        unsafe_allow_html=True)





# Visual 1: Price Distribution
st.subheader("Perfume Prices Distribution")
# Add a range slider for price
min_price, max_price = st.slider(
    'Select price range',
    min_value=float(all_perfume['price'].min()),
    max_value=float(all_perfume['price'].max()),
    value=(float(all_perfume['price'].min()), float(all_perfume['price'].max()))
)

# Filter the data based on the selected price range
filtered_data = all_perfume[(all_perfume['price'] >= min_price) & (all_perfume['price'] <= max_price)]

#Plot
fig, ax = plt.subplots()
sns.histplot(data=filtered_data, x='price', kde=True, bins=30, color='skyblue', ax=ax)
ax.set_title('Perfume Prices Distribution')
ax.set_xlabel('Price')
ax.set_ylabel('Number of Products')
st.pyplot(fig)

# Calculate and Display Summary Statistics
average_price = all_perfume['price'].mean()
highest_price = all_perfume['price'].max()
lowest_price = all_perfume['price'].min()



st.write(f"Average Price: ${average_price:.2f}")
st.write(f"Highest Price: ${highest_price:.2f}")
st.write(f"Lowest Price: ${lowest_price:.2f}")

# Add a description
st.write('- The chart shows the distribution of perfume prices with a long tail towards higher prices.')
st.write('- The majority of perfumes are priced below 100, with a peak around 50-100 price range.')




justified_text = """
<div style="text-align: justify;">
The perfume price distribution chart reveals a strong concentration of products in the lower price range, 
with the majority priced between $0 and $50. This indicates that the perfume market is highly competitive 
at lower price points, likely catering to price-sensitive consumers. The right-skewed distribution suggests 
that while luxury or high-end perfumes (priced above $100) exist, they represent a smaller segment of the 
market. This could highlight an opportunity for companies to introduce more premium products or create 
differentiated offerings to capture a niche but potentially profitable high-end segment. Additionally, 
businesses should focus on maintaining competitive pricing for mass-market products, as the bulk of consumer 
demand is likely in the affordable range.
<div>
"""
st.markdown(justified_text, unsafe_allow_html=True)

# Gender Analysis
st.subheader("Average Perfume Price by Gender")
average_price_men = all_perfume[all_perfume['gender'] == 'men']['price'].mean()
average_price_women = all_perfume[all_perfume['gender'] == 'women']['price'].mean()


fig, ax = plt.subplots()
ax.bar(['Men', 'Women'], [average_price_men, average_price_women], color=sns.color_palette('viridis', 2))
ax.set_title('Average Perfume Price by Gender')
ax.set_xlabel('Gender')
ax.set_ylabel('Average Price')
st.pyplot(fig)

st.write(f"Average Price for men perfume: ${average_price_men:.2f}")
st.write(f'Average price for women perfume: ${average_price_women:.2f}')

#Add description
justified_text = """
<div style="text-align: justify;">
This suggests that perfumes marketed towards and purchased. 
by men tend to have higher average pricing compared to perfumes targeted towards women.')
Counter to typical gender-based pricing (where women's products often cost more), 
men's perfumes command higher average prices. 
This challenges the 'pink tax'( assumption in the fragrance 
industry and suggests men might be less price-sensitive in this market.
Perfume brands may be able to command premium pricing for men's fragrances,
suggesting an opportunity to optimize pricing strategies based on gender.
<div>
"""
st.markdown(justified_text, unsafe_allow_html=True

# Add a title
st.subheader("Total Perfume Sales by Gender")

# Calculate the total sales and average price for each brand
brand_sales = all_perfume.groupby('brand')['sold'].sum()
brand_avg_price = all_perfume.groupby('brand')['price'].mean()

# Display total sales and average price for each brand
# Identify the top 10 brands by sales
top_brands = brand_sales.nlargest(10)

# Convert top_brands to a list to concatenate with "All" option
brand_options = ["All"] + top_brands.index.tolist()
brand_filter = st.selectbox("Select Brand", options=brand_options)

# Filter the DataFrame based on the selected brand
if brand_filter == "All":
    filtered_df = all_perfume
else:
    filtered_df = all_perfume[all_perfume['brand'] == brand_filter]

# Total Sales by Gender
total_sales_men = filtered_df[filtered_df['gender'] == 'men']['sold'].sum()
total_sales_women = filtered_df[filtered_df['gender'] == 'women']['sold'].sum()

# Calculate the total sales
total_sales = total_sales_men + total_sales_women

# Calculate the percentage sales
percent_sales_men = (total_sales_men / total_sales) * 100 if total_sales > 0 else 0
percent_sales_women = (total_sales_women / total_sales) * 100 if total_sales > 0 else 0

# Display bar chart
st.write('This bar chart shows the total perfume sales broken down by gender.')
fig, ax = plt.subplots()
ax.bar(['Men', 'Women'], [total_sales_men, total_sales_women], color=sns.color_palette('viridis', 2))
ax.set_title(f'Total Perfume Sales by Gender for {brand_filter}')
ax.set_xlabel('Gender')
ax.set_ylabel('Total Sales')
st.pyplot(fig)

# Display metrics with corrected quotation marks
st.write(f'Total Sales Men: ${total_sales_men:.2f}')
st.write(f'Total Sales Women: ${total_sales_women:.2f}')
st.write(f'- Men account for significantly higher total perfume sales compared to women for the brand {brand_filter}.')

# Create the pie chart
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['Men', 'Women']
sales_percentages = [percent_sales_men, percent_sales_women]
colors = sns.color_palette('viridis', len(labels))

ax.pie(sales_percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
ax.set_title(f'Total Perfume Sales by Gender (%) for {brand_filter}')
ax.axis('equal')

st.pyplot(fig)


# Add description
st.write(f'- The pie chart further illustrates the gender breakdown for the brand {brand_filter}.')

justified_text = """
<div style="text-align: justify;">

Perfume brands should consider tailoring their product offerings, marketing, and
sales strategies to better cater to the larger male customer base.

This indicates a stronger market demand or higher spending on fragrances targeted at men. 
The notable disparity suggests an opportunity for businesses to explore strategies to increase 
sales in the women's segment, such as targeted marketing campaigns, product innovation, or promotional 
efforts. The higher sales in the men's category may reflect differing consumer preferences or cultural 
factors that influence purchasing behavior, or it could imply that men are buying higher-priced products 
or that more products are being marketed to men. For companies, this insight could guide decisions on 
whether to continue focusing on the men's market or to diversify their offerings to better capture the 
potential in the women's segment.
</div>
"""
st.markdown(justified_text, unsafe_allow_html=True)


#Add title
st.subheader('Relationship between the top 10 brand popularity and pricing')

# Calculate the total sales and average price for each brand
brand_sales = all_perfume.groupby('brand')['sold'].sum()
brand_avg_price = all_perfume.groupby('brand')['price'].mean()

# Display total sales and average price for each brand
# Identify the top 10 brands by sales
top_brands = brand_sales.nlargest(10)
# Display the top 10 brands by sales
# Filter data for the top 10 brands
top_brand_avg_price = brand_avg_price[top_brands.index]
top_brand_sales = brand_sales[top_brands.index]

# Add a dropdown filter to select a brand
brand_options = ["All"] + list(top_brands.index)
selected_brand = st.selectbox("Select a Brand:", brand_options)

# Filter data based on selected brand
if selected_brand != "All":
    top_brand_avg_price = top_brand_avg_price[top_brand_avg_price.index == selected_brand]
    top_brand_sales = top_brand_sales[top_brand_sales.index == selected_brand]

# Visualize the relationship between the top 10 brand popularity and pricing with a bubble chart
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(
    x=top_brand_avg_price,
    y=top_brand_sales,
    size=top_brand_sales,
    sizes=(20, 2000),
    legend=False,
    alpha=0.6,
    ax=ax
)

# Annotate each point with the brand name
for brand in top_brand_avg_price.index:
    ax.text(
        x=top_brand_avg_price[brand],
        y=top_brand_sales[brand],
        s=brand,
        fontsize=9,
        ha='right',
        va='bottom'
    )

ax.set_title('Relationship Between Top 10 Brand Popularity and Pricing')
ax.set_xlabel('Average Price')
ax.set_ylabel('Total Sales')

st.pyplot(fig)
st.write('- There is a general positive correlation between a brands popularity (total sales) and its average price. ')

st.write( '- Calvin Klein and Versace are the most popular brands, '
           'with total sales over 100,000 and average prices around 30-35.')
justified_text = """
<div style="text-align: justify;">
    Having a strong brand value is essential, though its impact varies depending on
    the brand’s target audience and pricing strategy. For premium brands like Paco Rabanne, 
    brand value plays a critical role in attracting a niche market segment that is willing to pay 
    for exclusivity, luxury, and prestige, even if it results in lower overall sales. These brands 
    leverage their high-end image to justify premium prices, building loyalty among customers who 
    prioritize brand prestige over affordability. In contrast, affordable brands like Calvin Klein 
    also benefit significantly from strong brand value, as a reputable name helps them stand out in 
    a competitive market and attracts consumers seeking both value and a trusted brand. By positioning 
    themselves as high-quality yet accessible, these brands can capture a larger, price-sensitive audience, 
    achieving high sales volumes. Thus, brand value is important across all segments, though it’s leveraged differently: 
    premium brands use it to justify higher prices and exclusivity, while affordable brands use it to drive popularity and trust, 
    enhancing broad market appeal. This distinction underscores how brand value, when aligned with pricing strategy, can effectively 
    shape a brand’s market position and overall sales performance.
</div>
"""

st.markdown(justified_text, unsafe_allow_html=True)


st.subheader('Correlation between availability and sales')

# Filter out 'Unbranded' from the 'brand' column
valid_brands = [brand for brand in all_perfume['brand'].unique() if brand.lower() != 'unbranded']


# Visualize the correlation using a scatter plot

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=filtered_df, x='available', y='sold', alpha=0.6, ax=ax)

ax.set_title(f'Correlation Between Availability and Sales for {brand_filter}')
ax.set_xlabel('Availability')
ax.set_ylabel('Total Sales')

st.pyplot(fig)
# Calculate the correlation between availability and sales
availability_sales_corr = all_perfume['available'].corr(all_perfume['sold'])
st.write(f"Correlation between availability and sales: {availability_sales_corr:.2f}")

st.write('- Limited availability might create a sense of exclusivity')
st.write('- High-demand products might show as low availability due to selling out quickly')
st.write('- Products with very high availability might indicate less popular items that do not sell as well')

justified_text = """
<div style="text-align: justify;">
This is an interesting finding as it challenges the intuitive assumption that higher availability
would lead to higher sales. Instead, the data suggests that limited availability might be a
strategic advantage in the perfume market.

The scatter plot underscores the importance of demand-driven strategies over high inventory levels alone. 
For popular items with limited availability, high sales despite low stock suggest a "scarcity appeal" that 
may enhance consumer desire, positioning these items as exclusive or highly sought-after. Conversely, products
with high availability and low sales reveal the risks of overstocking, potentially reflecting poor demand 
forecasting or weak product-market fit. This imbalance highlights the need for brands to tailor their inventory 
strategies: high-demand items may benefit from more frequent restocking, while low-demand, overstocked items 
might require promotions, price adjustments, or strategic reductions to avoid excessive holding costs. 
Ultimately, the weak correlation between availability and sales suggests that brands should prioritize 
understanding consumer preferences and demand patterns to drive sales effectively, rather than relying 
solely on inventory levels.


</div>
"""
st.markdown(justified_text, unsafe_allow_html=True)


st.subheader("Feature Importance in Sales Prediction")

# Feature Importance Plot (with GradientBoostingRegressor)
# Define features and model pipeline
features = ['price', 'available', 'type', 'brand', 'gender']
X = all_perfume[features]
y = all_perfume['sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing and Model Pipeline
numeric_features = ['price', 'available']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_features = ['type', 'brand', 'gender']
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Feature selection
k = 100
feature_selector = SelectKBest(f_regression, k=k)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', feature_selector),
    ('regressor', GradientBoostingRegressor(n_estimators=450, random_state=42))
])

# Train and evaluate model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"Mean Absolute Error: {mae:.2f}")

# Extract Feature Importances
importance = model.named_steps['regressor'].feature_importances_
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
selected_features = feature_names[feature_selector.get_support()]

summed_importances = {
    'price': sum(importance[i] for i, f in enumerate(selected_features) if 'price' in f),
    'available': sum(importance[i] for i, f in enumerate(selected_features) if 'available' in f),
    'type': sum(importance[i] for i, f in enumerate(selected_features) if 'type' in f),
    'brand': sum(importance[i] for i, f in enumerate(selected_features) if 'brand' in f),
    'gender': sum(importance[i] for i, f in enumerate(selected_features) if 'gender' in f)
}

# Plot Feature Importances
fig, ax = plt.subplots()
ax.bar(summed_importances.keys(), summed_importances.values())
ax.set_xlabel('Features')
ax.set_ylabel('Summed Importance Score')
ax.set_title('Distribution of Summed Feature Importance')
st.pyplot(fig)

justified_text = """
<div style="text-align: justify;">
The chart highlights the distribution of feature importance in driving sales, 
with brand emerging as the most significant factor, followed by availability, price, type, 
and lastly, gender. This provides valuable insights for companies to refine their strategies. Brand 
building is paramount, as a strong brand identity heavily influences sales and customer loyalty. Investing 
in marketing, storytelling, and maintaining a positive reputation can help companies stand out in competitive 
markets. The high importance of **availability** suggests that stockouts negatively affect performance, 
underscoring the need for efficient supply chain management and accurate demand forecasting to align 
inventory with consumer needs. 

While price is less important than brand and availability, it still holds a notable influence, 
meaning businesses should carefully craft pricing strategies that align with their brand image, 
ensuring competitiveness without sacrificing profit margins. Additionally, the minimal importance 
of gender implies the product has broad, universal appeal, suggesting that companies can focus 
on inclusive marketing campaigns rather than over-segmenting their audience by gender. Finally, 
the moderate influence of product type signals an opportunity for companies to innovate and 
target popular or emerging categories, ensuring alignment with market trends without over-relying 
on niche offerings.

These insights provide actionable takeaways: prioritize brand development, ensure inventory 
optimization, strike the right balance in pricing, create inclusive marketing strategies, and 
focus product development on consumer trends. By aligning resources with these factors, businesses 
can effectively shape their market positioning and enhance overall sales performance.

</div>
"""
st.markdown(justified_text, unsafe_allow_html=True)


st.subheader('Geographical distribution of perfume sales')
# Use the relative path to the HTML file
html_file_path = 'perfume_sales_map.html'

# Check if the file exists before trying to open it
try:
    with open(html_file_path, "r") as file:
        html_content = file.read()

    # Display the HTML map in the Streamlit app
    components.html(html_content, height=600)

except FileNotFoundError:
    st.error(f"Error: The file '{html_file_path}' was not found.")

justified_text= """
<div style="text-align: justify;">
The geographical distribution of perfume sales reveals key market trends and opportunities. 
North America, particularly the United States and Canada, dominates with sales nearing 1.2 million units, 
reflecting a well-established market with strong consumer demand for fragrance products. Similarly, key Asian 
markets like China and India show significant sales, driven by large populations and a growing middle class. 
Western Europe, including perfume industry leaders like France, the UK, and Germany, also shows notable sales 
but at more moderate levels, highlighting these regions' long-standing affinity for luxury fragrances.

Emerging markets present exciting growth potential. Brazil stands out in South America with mid-range sales, 
while Russia and parts of the Middle East demonstrate increasing demand, signaling these regions' rising 
interest in luxury goods. However, many regions, particularly in Africa, Oceania, and parts of Eastern 
Europe, show low or no sales, highlighting untapped potential. Strategic focus here could involve 
tailored marketing efforts, building distribution networks, and fostering brand awareness to unlock 
future growth.

From a strategic perspective, companies should continue to prioritize high-sales regions like North 
America and Asia, where premium and diverse product offerings can capitalize on established consumer 
preferences. In emerging markets such as Brazil, India, and Russia, companies can focus on increasing 
market penetration through localized products or campaigns. Additionally, underdeveloped markets, 
especially in Africa, represent long-term opportunities where targeted awareness campaigns and 
expanding availability could drive growth as economies evolve.

In terms of consumer trends, the strong sales in North America and Western Europe suggest a mature market 
with preferences for premium and designer fragrances. These regions might benefit from exclusive product 
lines or limited editions to maintain consumer interest. In Asia, particularly in growing economies like 
China and India, the expanding middle class presents a promising opportunity for both luxury and mid-tier 
products. Ultimately, while high-sales regions should continue to be nurtured, strategic efforts should 
also aim at unlocking the potential in emerging and underdeveloped markets.
</div>
"""
st.markdown(justified_text, unsafe_allow_html=True)
