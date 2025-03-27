import polars as pl

data = pl.read_csv("../Coursework Data/Household data.csv")
data.describe()