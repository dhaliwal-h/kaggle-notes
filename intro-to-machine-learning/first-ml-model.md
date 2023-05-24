# First ML Model

Before building the model we first need to play around and see the data and try to make basic sense of what there is. For that we use pandas.

After reading the file into `DataFrame` we can use `DataFrame.columns` to see a list of column names in our data.

We can drop rows with missing values with

```python
melbourne_data = DataFrame.dropna(axis=0)
```

Pull out a single Column useing **dot-notation**
`DataFrame.Price` where `Price` is a column name.

```python
y = melbourne_data.Price
```

>`y` is named 'y' by convention and it contains our ***prediction target*** with `dtype` DataType of `Series` in `Pandas`

when we dont use all columns' data to build our models
we can choose particular columns called **features** like following

```python
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
```

Using bracket notation [] by convenvtion the variable is named `X` which is still of type `DataFrame`

>After dropping missing values and selectign particular features we get to ***Building Our Model***

As stated on kaggle website as follows are important definitions

You will use the `scikit-learn` library to create your models.

it is writtne `sklearn` while coding

The steps to building and using a model are:

1. **Define:** What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.

2. **Fit:** Capture patterns from provided data. This is the heart of modeling.

3. **Predict:** Just what it sounds like

4. **Evaluate:** Determine how accurate the model's predictions are.

For Example using `X` and `y` variables from above

```python
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)
```

>Many machine learning models allow some randomness in model training. Specifying a number for `random_state` ensures you get the same results in each run. This is considered a good practice. You use any number, and model quality won't depend meaningfully on exactly what value you choose.

Now we can use the `melbourne_model.predict()` function by passing New `DataFrame` of same size with same column names and predict unknown prices.
