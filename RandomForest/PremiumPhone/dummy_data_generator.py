import pandas as pd
import numpy as np
import random

def generate_dummy_data(n_train: int=500, n_test: int=200):
    def random_word_number():
        return random.choice(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])

    def generate_df(n: int):
        return pd.DataFrame({
            'p_id': np.arange(n),
            'Height(px)': [random_word_number() for _ in range(n)],
            'Width(px)': [random_word_number() for _ in range(n)],
            'RAM': [random.choice([512, 1024, 2048, 4, 6]) for _ in range(n)],
            'InternalStorage(MB or GB)': [random.choice([128, 256, 512, 1, 2, 4]) for _ in range(n)],
            'Weight(g or Kg)': [random.choice([0.2, 0.5, 150, 300]) for _ in range(n)],
            'target': [random.choice([0, 1]) for _ in range(n)] if n == n_train else None
        })

    train_df = generate_df(n_train)
    test_df = generate_df(n_test)
    test_df.drop(columns=['target'], inplace=True)

    return train_df, test_df
