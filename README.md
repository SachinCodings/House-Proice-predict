# House-Proice-predict
#how can we predict the price of the house or land.
#[houseprice machine learning.txt](https://github.com/user-attachments/files/16051425/houseprice.machine.learning.txt)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# Load the dataset
dataset = pd.read_csv('house_data.csv')

# Remove outliers using IQR method
def calc_elements_upper_and_lower_than_IQR(feature_name):
    Q1 = np.percentile(dataset[feature_name], 25, interpolation='midpoint')
    Q3 = np.percentile(dataset[feature_name], 75, interpolation='midpoint')
    IQR = Q3 - Q1
    upper_element_mask = dataset[feature_name] >= (Q3 + 1.5 * IQR)
    lower_element_mask = dataset[feature_name] <= (Q1 - 1.5 * IQR)
    return upper_element_mask, lower_element_mask

list_of_masks_for_outlier_removal = []
for feature in list_of_input_features_for_outliers:
    x, y = calc_elements_upper_and_lower_than_IQR(feature)
    list_of_masks_for_outlier_removal.append(x)
    list_of_masks_for_outlier_removal.append(y)

mask_for_outlier_removal_iqr = np.any(list_of_masks_for_outlier_removal, axis=0)
list_of_records_with_outliers_iqr = np.where(mask_for_outlier_removal_iqr)

dataset_clean_iqr = dataset.drop(list_of_records_with_outliers_iqr[0])

# Create polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
X_train_temp = dataset_clean_iqr.drop(['price'], axis=1)
poly_X_train = poly.fit_transform(X_train_temp)
poly_X_train = np.concatenate((poly_X_train, X_train_temp[:, -2:]), axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(poly_X_train, dataset_clean_iqr['price'], test_size=0.2, random_state=42)

# Create a linear regression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

# Create a ridge regression model
ridge_regression_model = Ridge(alpha=2500.0)
ridge_regression_model.fit(X_train, y_train)

# Make predictions
y_pred_linear = linear_regression_model.predict(X_test)
y_pred_ridge = ridge_regression_model.predict(X_test)

# Evaluate the models
print("Linear Regression Model:")
print("R squared score:", linear_regression_model.score(X_test, y_test))
print("Ridge Regression Model:")
print("R squared score:", ridge_regression_model.score(X_test, y_test))
