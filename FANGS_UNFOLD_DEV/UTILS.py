import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from matplotlib.colors import Normalize
from matplotlib import cm
from PIL import Image

class Utils:
    @staticmethod
    def create_directories(solver_type, output_dir, case_name):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/{case_name}_{solver_type.upper()}', exist_ok=True)

    @staticmethod
    def check_variables(defaults, scope):
        for var_name, default_value in defaults.items():
            if var_name not in scope:
                scope[var_name] = default_value

    @staticmethod
    def plot_1D_power(solver_type, data, x, g, output_dir=None, varname=None, case_name=None, title=None):
        plt.clf()

        plt.figure()
        plt.plot(x, data, 'g-', label=f'Group {g+1} - Magnitude - {varname}_{solver_type.upper()}')
        plt.legend()
        plt.ylabel('Normalized amplitude')
        plt.title(f'Magnitude G{g+1}')
        plt.xlabel('Distance from core centre [cm]')
        plt.grid()
        filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_{varname}_G{g+1}'
        plt.savefig(filename)
        plt.close()

        return filename

    @staticmethod
    def plot_1D_fixed(solver_type, data, x, g, output_dir=None, varname=None, case_name=None, title=None):
        plt.clf()

        plt.figure()
        plt.plot(x, np.abs(data)/np.max(np.abs(data)), 'g-', label=f'Group {g+1} - {varname}_{solver_type.upper()}')
        plt.legend()
        plt.ylabel('Normalized amplitude')
        plt.title(f'Magnitude G{g+1}')
        plt.xlabel('Distance from core centre [cm]')
        plt.grid()
        filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_magnitude_{varname}_G{g+1}'
        plt.savefig(filename)
        plt.close()

        plt.figure()
        plt.plot(x, np.degrees(np.angle(data)), 'g-', label=f'Group {g+1} - Phase - {varname}_{solver_type.upper()}')
        plt.legend()
        plt.ylabel('Normalized amplitude')
        plt.title(f'Magnitude G{g+1}')
        plt.xlabel('Distance from core centre [cm]')
        plt.grid()
        filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_phase_{varname}_G{g+1}'
        plt.savefig(filename)
        plt.close()

        return filename

    @staticmethod
    def plot_2D_rect_power(solver_type, data, x, y, g, cmap='viridis', output_dir=None, varname=None, case_name=None, title=None):
        plt.clf()

        extent = [x.min(), x.max(), y.min(), y.max()]
        plt.imshow(data, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')

        plt.colorbar()  # Add color bar to show scale
        if title:
            plt.title(title)
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')

        x_ticks = np.linspace(x.min(), x.max(), num=10)
        y_ticks = np.linspace(y.min(), y.max(), num=10)
        plt.xticks(x_ticks, labels=[f'{val:.1f}' for val in x_ticks])
        plt.yticks(y_ticks, labels=[f'{val:.1f}' for val in y_ticks])

        filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_{varname}_G{g}.png'
        plt.savefig(filename)
        plt.close()

        return filename

    @staticmethod
    def plot_2D_rect_fixed(solver_type, data, x, y, g, cmap='viridis', output_dir=None, varname=None, case_name=None, title=None, process_data=None):
        plt.clf()
        if process_data == 'magnitude':
            data = np.abs(data)  # Compute magnitude
        elif process_data == 'phase':
            data_rad = np.angle(data)  # Compute phase
            data = np.degrees(data_rad)  # Compute phase

        extent = [x.min(), x.max(), y.min(), y.max()]
        plt.imshow(data, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')

        if process_data == 'magnitude':
            plt.colorbar(label=f'{varname}{g}_mag')  # Add color bar to show scale
        elif process_data == 'phase':
            plt.colorbar(label=f'{varname}{g}_deg')  # Add color bar to show scale

        if title:
            plt.title(title)
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')

        x_ticks = np.linspace(x.min(), x.max(), num=10)
        y_ticks = np.linspace(y.min(), y.max(), num=10)
        plt.xticks(x_ticks, labels=[f'{val:.1f}' for val in x_ticks])
        plt.yticks(y_ticks, labels=[f'{val:.1f}' for val in y_ticks])

        filename = f'{output_dir}/{case_name}_{solver_type.upper()}/{case_name}_{solver_type.upper()}_{varname}_{process_data}_G{g}.png'
        plt.savefig(filename)
        plt.close()

        return filename

    @staticmethod
    def plot_2D_rect_fixed_general(solver_type, data, x, y, g, cmap='viridis', output=None, varname=None, case_name=None, title=None, process_data=None):
        plt.clf()
        if process_data == 'magnitude':
            data = np.abs(data)  # Compute magnitude
        elif process_data == 'phase':
            data_rad = np.angle(data)  # Compute phase
            data = np.degrees(data_rad)  # Compute phase

        extent = [x.min(), x.max(), y.min(), y.max()]
        plt.imshow(data, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')

        if process_data == 'magnitude':
            plt.colorbar(label=f'{varname}{g}_mag')  # Add color bar to show scale
        elif process_data == 'phase':
            plt.colorbar(label=f'{varname}{g}_deg')  # Add color bar to show scale

        if title:
            plt.title(title)
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')

        x_ticks = np.linspace(x.min(), x.max(), num=10)
        y_ticks = np.linspace(y.min(), y.max(), num=10)
        plt.xticks(x_ticks, labels=[f'{val:.1f}' for val in x_ticks])
        plt.yticks(y_ticks, labels=[f'{val:.1f}' for val in y_ticks])

        filename = f'{output}_{varname}_{process_data}_G{g}.png'
        plt.savefig(filename)
        plt.close()

        return filename
