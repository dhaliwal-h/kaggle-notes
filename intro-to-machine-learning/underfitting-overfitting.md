# Underfitting and Overfitting

> For a detailed explanation go to [Kaggle Link](https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting)

Sample Code

```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
```

Out put looks something like this

```txt
Max leaf nodes: 5            Mean Absolute Error:  347380
Max leaf nodes: 50           Mean Absolute Error:  258171
Max leaf nodes: 500          Mean Absolute Error:  243495
Max leaf nodes: 5000         Mean Absolute Error:  254983
```

> From above we can judge best value for us in this case is 500.

which we can use in `DecisionTreeRegresor(max_leaf_nodes = ** )` when building our model.
