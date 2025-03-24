import pandas as pd

def test_load_data():
    df = pd.read_csv('data/kaggle_datasets/body_fat/bodyfat.csv')
    assert not df.empty, "Loaded DataFrame is empty"
    assert 'BodyFat' in df.columns, "'BodyFat' column is missing"
