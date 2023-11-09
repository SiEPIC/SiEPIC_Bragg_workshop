import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def read_data(file_path):
    column_names = ['width', 'thickness', 'start', 'stop', 'n1', 'n2', 'n3']
    data = pd.read_csv(file_path, sep=',', header=None, names=column_names)
    
    return data

def plot_wg_data(data, lambda0=1550e-9):
    x = data['width']*1e9
    y = data['thickness']*1e9
    z = data['n1'] + data['n2']*(lambda0) + data['n3']*(lambda0)**2

    xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                         np.linspace(y.min(), y.max(), 100))
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    plt.pcolormesh(xi, yi, zi, shading='auto', cmap='viridis')
    plt.colorbar(label='Effective index')
    plt.xlabel('Waveguide Thickness [nm]')
    plt.ylabel('Waveguide Width [nm]')
    plt.title('Width vs Thickness vs Effective Index')
    plt.show()

if __name__ == "__main__":
    file_path = r'wg_data/wg_variability.txt'
    data = read_data(file_path)
    plot_wg_data(data, lambda0=1550e-9)
