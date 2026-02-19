Source: https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv

Dataset: Iris flower dataset
- Rows: 150
- Features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- Target: `species`

Quick practice with `DataSet`:
- `ds = DataSet.from_csv("data/iris.csv")`
- `X, y = ds.to_xy(target_column="species")`
- `train, test = ds.split(test_size=0.2, stratify_by="species")`
