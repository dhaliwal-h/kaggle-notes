# Intro To Machine Learning

>PREREQUISITE: You need to have working knowledge of python and programming.

A simple Machine Learning Model is **Decision Tree**

The step of capturing patterns from data is called **fitting** or **training** the model. The data used to fit the model is called the **training data**.

After the model has been fit, you can apply it to new data to predict prices of additional homes.

A decision tree that has more splits is called a deeper tree.

. The point at the bottom where we make a prediction is called a leaf.

The splits and values at the leaves will be determined by the data.

The first step in any machine learning process is getting familiar with the data which we do with pandas and import as such

```python
import pandas as pd
```

>Pandas is the primary tool data scientists use for exploring and manipulating data.

Use `pd.read_csv(stringFilePath)` to read a file of type `.csv` into a **DataFrame**

You can use `DataFrame.describe()` to describe the data loaded in the dataframe.
