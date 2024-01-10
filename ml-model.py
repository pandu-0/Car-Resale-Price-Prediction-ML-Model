# %% [markdown]
# # Car Resale Price Prediction using Scikit-Learn Testing

# %%
import pandas as pd

df = pd.read_csv("car_web_scraped_dataset.csv")

# %% [markdown]
# This is my first ever Notebook!!
# **I used these notebooks as reference while writing this notebook: [Ref1](https://www.kaggle.com/code/ayaz11/car-price-prediction#3.-EDA), [Ref2](https://www.kaggle.com/code/umutakba/car-prices-estimation)**

# %% [markdown]
# # Data Cleaning & Feature Engineering

# %%
# drop any duplicate rows in the dataframe
df_copy = df.copy().drop_duplicates()

# first strip the $ sign and then replace all occurances of a comma
# then convert the value to a integer
cleaned_price = [int(str(value).strip("$").replace(",", "")) for value in df_copy['price'].values]

print(cleaned_price)


# %% [markdown]
# We can see above that the price has been cleaned
# 
# Now let's clean the miles!

# %%
# In a similar fashion we clean the miles variable by replace commas and the word "miles", then stripping an trailing or leading spaces
# finally converting to an intiger
miles = [ int(str(m).replace("miles", "").replace(",", "").strip()) for m in df_copy["miles"].values ]

print(miles)
len(miles)

# %% [markdown]
# Now that our miles variable is clean, let's move onto to **name, color and condition** as these require splitting them into new coloumns:

# %%
temp_name = df_copy["name"].copy()

# split the column into 2 parts with the first being the make and the second being the model
temp_name = [str(name).split(" ", maxsplit=1) for name in temp_name.values]

# make of the car
make = [arr[0] for arr in temp_name]

print(make)
len(make)

# %%
# model of the car
model = [arr[1] for arr in temp_name]

print(model)
len(model)

# %% [markdown]
# With our name variable clean let's move onto **color**.

# %%
# Let's split this into 2 columns: exterior-color and interior-color
colors = [ str(color).split(",") for color in df_copy["color"].values ]

print(colors)
len(colors)

# %%
# Let's replace the word exterior and strip the string
exterior_color = [ str(color[0]).replace("exterior", "").strip() for color in colors ]

print(exterior_color)
len(exterior_color)

# %%
# Let's replace the word interior and strip the string
interior_color = [ str(color[1]).replace("interior", "").strip() for color in colors ]

print(interior_color)
len(interior_color)

# %% [markdown]
# Finally, let's deal with the **condition** variable that tells us important information about the **number of accidents** and the **amount of owners**

# %%
conditions = [ str(condition).split(",") for condition in df_copy['condition'].values ]

# replace the words in the column and make them an intiger
# we replace "No accidents reported" with a 0
accidents = [ int(str(condition[0]).replace("No", "0").replace("accidents reported", "").replace("accident reported", "").strip()) for condition in conditions]

print(accidents)
len(accidents)

# %%
# replace the words in the column and make them an intiger
owners = [ int(str(condition[1]).replace("Owners", "").replace("Owner", "").strip()) for condition in conditions]

print(owners)
len(owners)

# %%
# This is just to make sure that all the lists are the same size in order to make the dataframe
print(
    len(make), 
    len(model),
    len(df_copy['year'].copy()),
    len(miles),
    len(exterior_color),
    len(interior_color),
    len(accidents),
    len(owners),
    len(cleaned_price)
)

# %% [markdown]
# Now that we have cleaned all our columns, lets put them together into a clean dataframe!

# %%
# make the dataframe using our cleaned lists
df_clean = pd.DataFrame( 
    {
        "make": make,
        "model": model,
        "year": [int(yr) for yr in df_copy['year'].copy().values],
        "miles": miles,
        "exterior-color": exterior_color,
        "interior-color": interior_color,
        "accidents-reported": accidents,
        "num-of-owners": owners,
        "price": cleaned_price
    }
)

df_clean

# %% [markdown]
# A look at all the unique values in the dataset:

# %%
print(
    "make: {}".format(df_clean['make'].unique()),
    "model: {}".format(df_clean['model'].unique()),
    "year: {}".format(df_clean['year'].unique()),
    "miles: {}".format(df_clean['miles'].aggregate(func=['mean'])),
    "exterior-color: {}".format(df_clean['exterior-color'].unique()),
    "interior-color: {}".format(df_clean['interior-color'].unique()),
    "accidents-reported: {}".format(df_clean['accidents-reported'].unique()),
    "price: {}".format(df_clean['price'].aggregate(func=['mean'])),
    sep="\n\n"
)

# %% [markdown]
# As we can see above our dataset is now clean and ready for some data analysis!

# %% [markdown]
# # Data Analysis

# %%
df_clean.describe()

# %%
# Let's import matplotlib and seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(
    x="make", y="count", 
    data=df_clean['make'].value_counts().to_frame().head(10),
    hue="make"
)

plt.xticks(rotation=90);
plt.title("Top 10 Car Make Frequency");


# %% [markdown]
# Let's try to find out which variable has the **most** impact on the **price**!

# %%
df_clean

# %%
# a seaborn bargraph with each bar being a car make and the y-axis representing the avg price of a car of that make 
sns.barplot(
    x='make', y='price', 
    data=df_clean.groupby('make')['price'].mean().to_frame().sort_values(by='price', ascending=False).head(21),
    hue="make"
)


plt.title("Avg price by make");
plt.xticks(rotation=90);
plt.ylabel("Average Price - part 1");

# %%
# a seaborn bargraph with each bar being a car make and the y-axis representing the avg price of a car of that make
sns.barplot(
    x='make', y='price',
    data=df_clean.groupby('make')['price'].mean().to_frame().sort_values(by='price', ascending=False).tail(21),
    hue="make",
    
)
plt.title("Avg price by make - part 2");
plt.xticks(rotation=90);
plt.ylabel("Average Price");

# %% [markdown]
# Now let's see if there's any correlation between accidents reported and number of previous owners!
# 
# But before we do that, let's first remove some outliers.

# %%
# simple seaborn box plot by using the price as datapoints
box = sns.boxplot(
    x=df_clean["price"].sort_values()
)

plt.show()

print(df_clean["price"].sort_values().describe())

# %% [markdown]
# We can see above that the **1st quartile** is **17991** and **3rd quartile is 22999**.
# 
# Let's use these to remove any outliers.
# 
# Here's a graph before removing them:

# %%
# grouby accidents-reported and aggregate price using mean
sns.barplot(
    x='accidents-reported', y='price',
    data=df_clean.groupby('accidents-reported')['price'].mean().to_frame(),
    hue="accidents-reported"
)

plt.title("mean price by accidents-reported")

# %% [markdown]
# Here we can easily notice a correlation between the number of accidents reported and the mean price of the car, but let's look at it again
# after removing outliers.

# %%
# function that calculates the lower and upper bounds for finding outliers using the Inter Quartile Range (IQR)
def return_lower_upper_bounds(q1: float, q3: float):
    IQR = q3 - q1
    lower = q1 - (1.5 * IQR)
    upper = q3 + (1.5 * IQR)
    return lower, upper

# make a copy of the dataframe
df_removed_outliers_by_price = df_clean.copy()

# use the method we defined previously to remove outliers
lower_bound, upper_bound = return_lower_upper_bounds(q1=17991, q3=30999.25)

# find the indexes of the rows that have price outliers
index_q1 = df_removed_outliers_by_price[ (df_removed_outliers_by_price['price'] < lower_bound) ].index
index_q3 = df_removed_outliers_by_price[ (df_removed_outliers_by_price['price'] > upper_bound) ].index

# remove them from our dataframe
df_removed_outliers_by_price.drop(index=index_q1, inplace=True)
df_removed_outliers_by_price.drop(index=index_q3, inplace=True)

df_removed_outliers_by_price

# %%
sns.boxplot(x=df_removed_outliers_by_price['price'].sort_values())

# %% [markdown]
# Above we can see that we have successfully **removed** the **outliers** (I think some are still there because seaborn calculates them differently but we have removed the **extreme** ones so we're safe)! 
# 
# Now, let's make the bargraph again.

# %%

sns.barplot(
    x='accidents-reported', y='price', 
    data=df_removed_outliers_by_price.groupby('accidents-reported')['price'].mean().to_frame(),
    hue='accidents-reported',
    legend=False
)
plt.title("mean price by accidents-reported");


# %% [markdown]
# Now we can observe a more **direct correlation** between **accidents-reported** and the **mean** price in the above bargraph.
# 
# Naturally, the **lower**<span>&darr;</san> amount of **accidents reported** for a car, the **more**<span>&uarr;</san> the value as there's **less damaged** that has been done to the car.
# 
# Let's do this for the owners variable as well!

# %% [markdown]
# Again, this is **before** removing the outliers:

# %%
# grouby accidents-reported and aggregate price using mean
sns.barplot(
    x='num-of-owners', y='price', 
    data=df_clean.groupby('num-of-owners')['price'].mean().to_frame(),
    hue='num-of-owners',
    legend=False
)

# set the plot's title
plt.title("mean price by accidents-reported");

df_clean

# %% [markdown]
# This is **after** removing the outliers:

# %%
sns.barplot(
    x='num-of-owners', y='price', 
    data=df_removed_outliers_by_price.groupby('num-of-owners')['price'].mean().to_frame(),
    hue="num-of-owners",
    legend=False
)

plt.title("mean price by accidents-reported");

# %% [markdown]
# Seems like the outliers had **little** to no effect on the number of previous owners!
# 
# It is also worth noticing that some of these cars have 0 previous owners which does not make sense because this data is off of a dataset that claims that these are used car listings.
# 
# Let's find out how many of these cars have 0 owners.

# %%
print(df_clean['num-of-owners'].value_counts()[0])

# %% [markdown]
# Seems like we only have 10 rows that have 0 as the number of previous owners. So let's drop those rows to maintain our datasets integrity.

# %%
# grab the indices of the rows that have 0 as the number of previous owners
index_of_new_cars = df_clean[(df_clean['num-of-owners'] == 0)].index

df_clean.drop(index=index_of_new_cars, inplace=True)

temp  = list(df_clean['num-of-owners'].unique())
temp.sort()
print(temp)

# %% [markdown]
# As we can see, there's **no 0** in the above list because we have **eliminated** it.

# %% [markdown]
# Let's also drop them from the no outliers dataframe:

# %%
index_of_new_cars = df_removed_outliers_by_price[(df_removed_outliers_by_price['num-of-owners'] == 0)].index

df_removed_outliers_by_price.drop(index=index_of_new_cars, inplace=True)

# %% [markdown]
# Now let's take a look the most **important** relationship which is between **miles** and **price**!
# 
# Let's visualize this using a scatter plot!!

# %%
sns.scatterplot(
    x='miles', y='price', 
    data=df_removed_outliers_by_price, hue='year' , s=25, alpha=0.5
)

# %% [markdown]
# The plot above is too dense and hard to read, so I decided to make the figure larger and lengthen the yticks.

# %%
# set figure size
plt.figure(figsize=(10,7))
# scatter plot of miles vs price with hue to the year variable, "s" param is the size of the dots & alpha is the opacity
sns.scatterplot(
    x='miles', y='price', data=df_removed_outliers_by_price, 
    hue='year', s=25, alpha=0.65
).set_yticks(range(0, 55000, 5000)) # set yticks range by giving it a list spaced by 5000

# %% [markdown]
# From the above plot, we can conclude that **newer** cars are **driven less**, hence the **higher price**<span>&uarr;</san>. And **older cars** are driven more, because well they're old, so they **cost less**<span>&darr;</san>.
# 
# In the middle of scatter plot, however, we can see a fair mix of both **old** and **new** cars. This is also where our mean price is.

# %%
plt.figure(figsize=(10,10))
# scatter plot and then a density plot over it
# hue is to give color by a variable, and alpha is opacity of the points
sns.relplot(
    x='miles', y='price', data=df_removed_outliers_by_price, 
    hue='year' , s=25, alpha=0.65
) # scatter plot

# levels is the amount of lines in the density plot
sns.kdeplot(
    x='miles', y='price', data=df_removed_outliers_by_price, 
    alpha=0.5, levels=5, color="black"
) # density plot over scatter plot

plt.title("miles vs price scatter plot");

# %% [markdown]
# Finally, let's take quick look at **exterior** and **interior** color prices.

# %%
ex_df = df_clean.groupby('exterior-color')['price'].mean().sort_values(ascending=False).to_frame().sort_index()
# add the count column
ex_df['count'] = df_clean['exterior-color'].value_counts().to_frame().sort_index()

sns.barplot(
    x="exterior-color", y="price", 
    data=ex_df.sort_values(by="price", ascending=False), 
    # rocket_r is the predefined magma color palette reversed
    hue="count", palette=sns.color_palette("rocket_r", as_cmap=True) # as_cmap is "as continuous color map"
)

plt.title("exterior color avg price with color count");
plt.xticks(rotation=90);

# %%
# groupby interior color and aggregate by finding the mean price
# we sort index of both the dataframes in_df and df_clean so that their rows match up correctly when joining them
in_df = df_clean.groupby('interior-color')['price'].mean().sort_values(ascending=False).to_frame().sort_index()

# add the count column
in_df['count'] = df_clean['interior-color'].value_counts().to_frame().sort_index()

sns.barplot(
    x="interior-color", y="price", 
    data=in_df.sort_values(by="price", ascending=False), 
    # rocket_r is the predefined magma color palette reversed
    hue="count", palette=sns.color_palette("rocket_r", as_cmap=True) # as_cmap is "as continuous color map"   
)

plt.title("interior color avg price with color count");
plt.xticks(rotation=90);

# %% [markdown]
# Let's also check how often these colors occur in our dataset.

# %% [markdown]
# **White** and **Black** are the most **common exterior colors**. **Black** and **Grey** are the **most common interior colors**.
# 
# From the count colors, we can infer that since the colors like <span style="color:orange">**Orange**</span> and <span style="color:Green">**Green**</span> have a **very low**<span>&darr;</span> count, they **cost more**<span>&uarr;</span> and since **Black** is a **common** color<span>&uarr;</span> so it comes at a **lower** price<span>&darr;</span>.

# %% [markdown]
# # Preprocessing

# %% [markdown]
# Let's preprocessing the data!

# %%
# import StandardScaler, QuantileScaler, and One Hot Encoder
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
# import ColumnTransformer to combine scaling and encoding
from sklearn.compose import ColumnTransformer
# import Pipeline
from sklearn.pipeline import Pipeline
# import train_test_split
from sklearn.model_selection import train_test_split

# a copy of our dataframe
df_final = df_removed_outliers_by_price.copy()

# df_final

# features: make, model, year, miles, exterior-color. interior-color, accidents-reported, and num-of-owners
X = df_final.drop(columns=['price'])
# labels: price
y = pd.DataFrame(df_final['price'].copy(), columns=["price"])

# %%
cols = ['year', 'miles', 'accidents-reported', 'num-of-owners', 'price']

before = df_final[cols].describe()

after = pd.DataFrame(StandardScaler().fit_transform(df_final[cols]), columns=cols)

print(
    "brefore:", 
    before.describe(),
    sep="\n\n"
)
print()
print(
    "after:",
    after.describe(),
    sep="\n\n"
)


# %% [markdown]
# A look at the effects of the **StandardScaler** to scale our data.

# %%
cols = ['year', 'miles', 'accidents-reported', 'num-of-owners', 'price']

before = df_final[cols].describe()

after = pd.DataFrame(QuantileTransformer().fit_transform(df_final[cols]), columns=cols)

print(
    "brefore:", 
    before.describe(),
    sep="\n\n"
)
print()
print(
    "after:",
    after.describe(),
    sep="\n\n"
)

# %% [markdown]
# Above we can see that the QuantileTransformer gives us a more easier scale to work with by avoding negative values

# %% [markdown]
# Correltion before the QuantileTransformer

# %%
corr = before.corr()

abs(corr["price"]).sort_values(ascending=False)

# %% [markdown]
# Correltion after the QuantileTransformer

# %%
# use the correlation method to see the relation between numerical variables
corr = after.corr()

abs(corr['price']).sort_values(ascending=False)

# %% [markdown]
# Above we can notice that the effect of **miles** and **accidents-reported** has **decreased** and the effect of **year** and **num-of-owners** has significantly **increased**.

# %%
import numpy as np
from sklearn.impute import SimpleImputer

# Initialize the SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imp.fit_transform(X)

# %%
# we use the ColumnTransformer to scale and encode
preprocessor = ColumnTransformer(
    transformers=[
        # numerical columns scaled using the QuantileTransformer
        ("num", QuantileTransformer(), ['year', 'miles', 'accidents-reported', 'num-of-owners']),
        # categorical columns encoded using the OneHotEncoder
        ("cat", OneHotEncoder(handle_unknown='ignore'),  ['make', 'model', 'exterior-color', 'interior-color'])
    ]
)

# fit the data
# X_transformed = preprocessor.fit_transform(X)
# y_transformed = ColumnTransformer(transformers=[ ('num', QuantileTransformer(), ['price']) ]).fit_transform(y)

# %% [markdown]
# # ML Pipeline

# %%
# split the data into Train and Test datasets with the test_size being 20% of the original dataset
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

# import models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import precision_score, f1_score, make_scorer
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import LogisticRegression

# Contruct our pipline
RFR = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=5))
])

KNN = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor())
])


# %% [markdown]
# ## GridSearch

# %%
forest_model = GridSearchCV(
    estimator=RFR,
    # max_features: It decides how many features each tree in the RF considers at each split
    param_grid={
        'model__bootstrap': [True, False],
        'model__criterion': ['friedman_mse', 'squared_error', 'poisson'],
        # 'model__max_depth': [2, 4],
        # 'model__max_features': [3, 4, 5, 6, 7],
        # 'model__min_samples_leaf':[1,2],
        # 'model__min_samples_split': [2,5],
        'model__n_estimators': [int(x) for x in np.linspace(10, 80, 10)],
    },
    # scoring={"precision": make_scorer(precision_score)},
    verbose=2,
    n_jobs=4,
    # so that our model is refit with the best params
    refit=True,
    cv=3
)

knn_model = GridSearchCV(
    estimator=KNN,
    # max_features: It decides how many features each tree in the RF considers at each split
    param_grid={
        'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'model__weights': ['uniform', 'distance']
    },
    verbose=2,
    n_jobs=4,
    # so that our model is refit with the best params
    refit=True,
    cv=3
)

# RFR.get_params()
# KNN.get_params()

# %% [markdown]
# # Model Training

# %% [markdown]
# ## RandomForestRegressor

# %%
forest_model.fit(X_tr, np.ravel(y_tr))

# %%
forest_model.best_params_

# %% [markdown]
# ## KNeighborsRegressor

# %%
knn_model.fit(X_tr, np.ravel(y_tr))

# %% [markdown]
# ## Results

# %% [markdown]
# Accuracy of RandomForestRegressor:

# %%
print("Train: ", forest_model.score(X_tr, np.ravel(y_tr)))
print("Test: ", forest_model.score(X_te, np.ravel(y_te)))

# %% [markdown]
# Accuracy of KNearestNeighborsRegressor:

# %%
print("Train", knn_model.score(X_tr, np.ravel(y_tr)))
print("Test", knn_model.score(X_te, np.ravel(y_te)))

# %% [markdown]
# This is an overfit! ^^

# %%
pd.DataFrame(forest_model.cv_results_).head()

# %%
pd.DataFrame(knn_model.cv_results_).head()

# %%
sns.scatterplot(x=knn_model.predict(X_te), y=np.ravel(y_te))
plt.title("predicted vs actual ")
plt.xlabel("predicted")
plt.ylabel("actual")

# %%
pd.DataFrame(forest_model.cv_results_).head()

# %% [markdown]
# Most Accurate Model we have:

# %%
sns.scatterplot(x=forest_model.predict(X_te), y=np.ravel(y_te))
plt.title("RandomForest: predicted vs actual ")
plt.xlabel("predicted")
plt.ylabel("actual")

# %% [markdown]
# ## Sample Predictions

# %%
# This is my car, actual price is around 11k
myCar = pd.DataFrame(
    {   "make": "Toyota",
        "model": "Corolla",
        "year": 2015,
        "miles": 90000,
        "exterior-color": "Black",
        "interior-color": "Gray",
        "accidents-reported": 2,
        "num-of-owners": 3
    }, 
    index=[0]
)

# cars I found online, for this one the actual price is 24k
merc = pd.DataFrame(
    {   "make": "Mercedes-Benz",
        "model": "S-Class",
        "year": 2014,
        "miles": 89000,
        "exterior-color": "White",
        "interior-color": "Beige",
        "accidents-reported": 0,
        "num-of-owners": 1
    }, 
    index=[0]
)

# This one is 17K
ford = pd.DataFrame(
    {   "make": "Ford",
        "model": "F-150",
        "year": 2013,
        "miles": 121000,
        "exterior-color": "Gray",
        "interior-color": "Gray",
        "accidents-reported": 0,
        "num-of-owners": 1
    }, 
    index=[0]
)

print(
    f"KNearestNeighbors: Your car resale price is: ${round(knn_model.predict(X=myCar)[0], 2)}"
)

print(
    f"RandomForestRegressor: Your car resale price is: ${round(forest_model.predict(X=myCar)[0], 2)}"
)

# %% [markdown]
# # End


