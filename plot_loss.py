import pandas as pd
import matplotlib.pyplot as plt
from params import Params
import numpy as np

if __name__ == '__main__':
    params = Params()
    model_name = params.model_name

    loss_df = pd.read_csv(f'log/{model_name}/history.csv')

    # loss_df.plot()
    plt.Figure()
    plt.plot(loss_df['epoch'], loss_df['d_loss'], label="d_loss")
    # plt.plot(loss_df['epoch'], loss_df['g_loss'], label="g_loss")

    plt.xlabel('Epoch')
    plt.legend()

    plt.show()
