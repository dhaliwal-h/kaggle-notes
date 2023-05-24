# Model Validation

Model Validation is to measure the quality of model by checking the accuracy of our predictions.

>***DONOT*** Make predictions with the data you used to train your model.

To meassure Model Quality we use **Mean Absolute Error** `error=actual-predicted`

To use it we use

```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)

mean_absolute_error(y, predicted_home_prices)
```

The measure we just computed can be called an "in-sample" score. We used a single "sample" of houses for both building the model and evaluating it.

>This is what we should not do.

Since models' practical value come from making predictions on new data, we measure performance on data that wasn't used to build the model. The most straightforward way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. This data is called ***validation data***

> Here is how to do the above mentioned

```python
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
```
