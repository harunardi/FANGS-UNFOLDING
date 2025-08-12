import numpy as np
from scipy.sparse import lil_matrix
import os
import sys
import h5py
from scipy.interpolate import RBFInterpolator, griddata

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

# Function to convert 1D indexes
def convert_index_1D(D, N):
    """
    Convert 1D index mapping based on nonzero elements in D.
    
    Parameters:
        D (array): 2D array where D[0][n] is checked for nonzero.
        N (int): Number of elements.
    
    Returns:
        list: Conversion array with incremented indices for nonzero D[0][n].
    """
    conv = [0] * (N)
    tmp_conv = 0
    for n in range(N):
        if D[0][n] != 0:
            tmp_conv += 1
            conv[n] = tmp_conv
    return conv

# Function to save data in HDF5 format
def save_output_hdf5(filename, output_dict):
    """
    Save output dictionary with complex numbers to HDF5 file.
    
    Parameters:
        filename (str): Output HDF5 file path.
        output_dict (dict): Dictionary with keys and list of dicts with 'real' and 'imaginary'.
    """
    with h5py.File(filename, 'w') as f:
        for key, value in output_dict.items():
            real_data = np.array([complex_number['real'] for complex_number in value])
            imag_data = np.array([complex_number['imaginary'] for complex_number in value])
            f.create_dataset(f'{key}/real', data=real_data)
            f.create_dataset(f'{key}/imaginary', data=imag_data)

# Function to save sparse matrix to file
def save_sparse_matrix(A, filename):
    """
    Save a sparse matrix to a text file in COO format.
    
    Parameters:
        A (scipy.sparse matrix): Sparse matrix to save.
        filename (str): Output file path.
    """
    A_coo = A.tocoo()
    I, J, V = A_coo.row, A_coo.col, A_coo.data
    
    with open(filename, 'w') as file:
        for i, j, v in zip(I, J, V):
            file.write(f"{i} {j} {v}\n")
    
    print(f"Sparse matrix saved to {filename}")

##############################################################################
def FORWARD_D_1D_matrix(group, BC, N, dx, D):
    """
    Construct the forward diffusion matrix for 1D geometry with boundary conditions.
    
    Parameters:
        group (int): Number of energy groups.
        BC (list): Boundary conditions [left, right].
        N (int): Number of spatial cells.
        dx (float): Cell width.
        D (array): Diffusion coefficients.
    
    Returns:
        lil_matrix: Sparse matrix for forward diffusion.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))

    # Initialize BC
    BC_left = BC[0]
    BC_right = BC[1]
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            if n == 0:
                if BC_left == 1:  # Zero Flux
                    matrix[g*N, g*N] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1]))) +((2 * D[g][0]) / (dx**2))
                    matrix[g*N, g*N+1] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                elif BC_left == 2:  # Reflective
                    matrix[g*N, g*N] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                    matrix[g*N, g*N+1] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                elif BC_left == 3:  # Vacuum
                    matrix[g*N, g*N] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1]))) +((2 * D[g][0]) /((4*D[g][0]*dx)+(dx**2)))
                    matrix[g*N, g*N+1] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                else:
                    raise ValueError("Invalid BC")
            elif n == N - 1:
                if BC_right == 1:  # Zero Flux
                    matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += ((2 * D[g][n]) / (dx**2)) +((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                elif BC_right == 2:  # Reflective
                    matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += ((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                elif BC_right == 3:  # Vacuum
                    matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += ((2 * D[g][n]) /((4*D[g][n]*dx)+(dx**2))) +((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                else:
                    raise ValueError("Invalid BC")
            else:
                matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                matrix[g*N+n, g*N+n] += (((2 * D[g][n+1]*D[g][n] / ((dx**2) * (D[g][n + 1]+D[g][n]))) + (2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n]))))
                matrix[g*N+n, g*N+n + 1] += -((2 * D[g][n + 1]*D[g][n]) / ((dx**2) * (D[g][n + 1]+D[g][n])))

    return matrix

def FORWARD_NUFIS_1D_matrix(group, N, chi, NUFIS):
    """
    Construct the forward fission production matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        chi (array): Fission spectrum.
        NUFIS (array): Nu-fission cross sections.
    
    Returns:
        lil_matrix: Sparse matrix for fission production.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N,group*N))
    for i in range(group):
        for j in range(group):
            for k in range(N):
                matrix[i*N + k, j*N + k] = chi[i][k]*NUFIS[j][k]

    return matrix

def FORWARD_SCAT_1D_matrix(group, N, SIGS):
    """
    Construct the forward scattering matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        SIGS (array): Scattering cross sections.
    
    Returns:
        lil_matrix: Sparse matrix for scattering.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))

    if group == 1:
        for i in range(N):
            matrix[i, i] = SIGS[0][i]
    else:
        for i in range(group):
            for j in range(group):
                for k in range(N):
                    matrix[i * N + k, j * N + k] += SIGS[i][j][k]

    return matrix

def FORWARD_TOT_1D_matrix(group, N, TOT):
    """
    Construct the forward total cross section matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        TOT (array): Total cross sections.
    
    Returns:
        lil_matrix: Sparse matrix for total cross section.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            matrix[g*N+n, g*N+n] += TOT[g][n]

    return matrix

##############################################################################
def ADJOINT_D_1D_matrix(group, BC, N, dx, D):
    """
    Construct the adjoint diffusion matrix for 1D geometry with boundary conditions.
    
    Parameters:
        group (int): Number of energy groups.
        BC (list): Boundary conditions [left, right].
        N (int): Number of spatial cells.
        dx (float): Cell width.
        D (array): Diffusion coefficients.
    
    Returns:
        lil_matrix: Sparse matrix for adjoint diffusion.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))

    # Initialize BC
    BC_left = BC[0]
    BC_right = BC[1]
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            if n == 0:
                if BC_left == 1:  # Zero Flux
                    matrix[g*N, g*N] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1]))) +((2 * D[g][0]) / (dx**2))
                    matrix[g*N, g*N+1] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                elif BC_left == 2:  # Reflective
                    matrix[g*N, g*N] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                    matrix[g*N, g*N+1] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                elif BC_left == 3:  # Vacuum
                    matrix[g*N, g*N] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1]))) +((2 * D[g][0]) /((4*D[g][0]*dx)+(dx**2)))
                    matrix[g*N, g*N+1] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                else:
                    raise ValueError("Invalid BC")
            elif n == N - 1:
                if BC_right == 1:  # Zero Flux
                    matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += ((2 * D[g][n]) / (dx**2)) +((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                elif BC_right == 2:  # Reflective
                    matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += ((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                elif BC_right == 3:  # Vacuum
                    matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += ((2 * D[g][n]) /((4*D[g][n]*dx)+(dx**2))) +((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                else:
                    raise ValueError("Invalid BC")
            else:
                matrix[g*N+n, g*N+n - 1] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                matrix[g*N+n, g*N+n] += (((2 * D[g][n+1]*D[g][n] / ((dx**2) * (D[g][n + 1]+D[g][n]))) + (2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n]))))
                matrix[g*N+n, g*N+n + 1] += -((2 * D[g][n + 1]*D[g][n]) / ((dx**2) * (D[g][n + 1]+D[g][n])))

    return matrix

def ADJOINT_TOT_1D_matrix(group, N, TOT):
    """
    Construct the adjoint total cross section matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        TOT (array): Total cross sections.
    
    Returns:
        lil_matrix: Sparse matrix for adjoint total cross section (transposed).
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            matrix[g*N+n, g*N+n] += TOT[g][n]

    return matrix.transpose()

def ADJOINT_SCAT_1D_matrix(group, N, SIGS):
    """
    Construct the adjoint scattering matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        SIGS (array): Scattering cross sections.
    
    Returns:
        lil_matrix: Sparse matrix for adjoint scattering (transposed).
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))

    if group == 1:
        for i in range(N):
            matrix[i, i] = SIGS[0][i]
    else:
        for i in range(group):
            for j in range(group):
                for k in range(N):
                    matrix[i * N + k, j * N + k] += SIGS[i][j][k]

    return matrix.transpose()

def ADJOINT_NUFIS_1D_matrix(group, N, chi, NUFIS):
    """
    Construct the adjoint fission production matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        chi (array): Fission spectrum.
        NUFIS (array): Nu-fission cross sections.
    
    Returns:
        lil_matrix: Sparse matrix for adjoint fission production (transposed).
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N,group*N))
    for i in range(group):
        for j in range(group):
            for k in range(N):
                matrix[i*N + k, j*N + k] = chi[i][k]*NUFIS[j][k]

    return matrix.transpose()

##############################################################################
def NOISE_D_1D_matrix(group, BC, N, dx, D):
    """
    Construct the noise diffusion matrix for 1D geometry with boundary conditions.
    
    Parameters:
        group (int): Number of energy groups.
        BC (list): Boundary conditions [left, right].
        N (int): Number of spatial cells.
        dx (float): Cell width.
        D (array): Diffusion coefficients.
    
    Returns:
        lil_matrix: Sparse matrix for noise diffusion.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N))

    # Initialize BC
    BC_left = BC[0]
    BC_right = BC[1]
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            if n == 0:
                if BC_left == 1:  # Zero Flux
                    matrix[g*N, g*N] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1]))) -((2 * D[g][0]) / (dx**2))
                    matrix[g*N, g*N+1] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                elif BC_left == 2:  # Reflective
                    matrix[g*N, g*N] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                    matrix[g*N, g*N+1] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                elif BC_left == 3:  # Vacuum
                    matrix[g*N, g*N] += -((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1]))) -((2 * D[g][0]) /((4*D[g][0]*dx)+(dx**2)))
                    matrix[g*N, g*N+1] += ((2 * D[g][0]*D[g][1]) / ((dx**2) * (D[g][0]+D[g][1])))
                else:
                    raise ValueError("Invalid BC")
            elif n == N - 1:
                if BC_right == 1:  # Zero Flux
                    matrix[g*N+n, g*N+n - 1] += ((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += -((2 * D[g][n]) / (dx**2)) -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                elif BC_right == 2:  # Reflective
                    matrix[g*N+n, g*N+n - 1] += ((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                elif BC_right == 3:  # Vacuum
                    matrix[g*N+n, g*N+n - 1] += ((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                    matrix[g*N+n, g*N+n] += -((2 * D[g][n]) /((4*D[g][n]*dx)+(dx**2))) -((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                else:
                    raise ValueError("Invalid BC")
            else:
                matrix[g*N+n, g*N+n - 1] += ((2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n])))
                matrix[g*N+n, g*N+n] += -(((2 * D[g][n+1]*D[g][n] / ((dx**2) * (D[g][n + 1]+D[g][n]))) + (2 * D[g][n - 1]*D[g][n]) / ((dx**2) * (D[g][n - 1]+D[g][n]))))
                matrix[g*N+n, g*N+n + 1] += ((2 * D[g][n + 1]*D[g][n]) / ((dx**2) * (D[g][n + 1]+D[g][n])))

    return matrix

def NOISE_TOT_1D_matrix(group, N, TOT):
    """
    Construct the noise total cross section matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        TOT (array): Total cross sections.
    
    Returns:
        lil_matrix: Sparse matrix for noise total cross section.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N), dtype=complex)
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            matrix[g*N+n, g*N+n] += TOT[g][n]

    return matrix

def NOISE_SCAT_1D_matrix(group, N, SIGS):
    """
    Construct the noise scattering matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        SIGS (array): Scattering cross sections.
    
    Returns:
        lil_matrix: Sparse matrix for noise scattering.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N), dtype=complex)

    if group == 1:
        for i in range(N):
            matrix[i, i] = SIGS[0][i]
    else:
        for i in range(group):
            for j in range(group):
                for k in range(N):
                    matrix[i * N + k, j * N + k] += SIGS[i][j][k]

    return matrix

def NOISE_NUFIS_1D_matrix(group, N, chi_p, chi_d, NUFIS, k_complex, Beff, keff):
    """
    Construct the noise fission production matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        chi_p (array): Prompt fission spectrum.
        chi_d (array): Delayed fission spectrum.
        NUFIS (array): Nu-fission cross sections.
        k_complex (complex): Complex eigenvalue.
        Beff (float): Effective delayed neutron fraction.
        keff (float): Effective multiplication factor.
    
    Returns:
        lil_matrix: Sparse matrix for noise fission production.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N,group*N), dtype=complex)
    for i in range(group):
        for j in range(group):
            for k in range(N):
                matrix[i*N + k, j*N + k] = (chi_p[i][k] * (1-Beff)/keff + chi_d[i][k] * k_complex) * NUFIS[j][k]

    return matrix

def NOISE_FREQ_1D_matrix(group, N, omega, v):
    """
    Construct the noise frequency matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        omega (float): Frequency.
        v (array): Neutron velocities.
    
    Returns:
        lil_matrix: Sparse matrix for noise frequency.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N), dtype=complex)
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            matrix[g*N+n, g*N+n] += 1j*omega/v[g][n]

    return matrix

def NOISE_dTOT_1D_matrix(group, N, dTOT):
    """
    Construct the noise perturbed total cross section matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        dTOT (array): Perturbed total cross sections.
    
    Returns:
        lil_matrix: Sparse matrix for noise perturbed total cross section.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N), dtype=complex)
    
    # Create tridiagonal matrix for upper left quadrant with boundary conditions
    for g in range(group):
        for n in range(N):
            matrix[g*N+n, g*N+n] += dTOT[g][n]

    return matrix

def NOISE_dSCAT_1D_matrix(group, N, dSIGS):
    """
    Construct the noise perturbed scattering matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        dSIGS (array): Perturbed scattering cross sections.
    
    Returns:
        lil_matrix: Sparse matrix for noise perturbed scattering.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N, group*N), dtype=complex)

    if group == 1:
        for i in range(N):
            matrix[i, i] = dSIGS[0][i]
    else:
        for i in range(group):
            for j in range(group):
                for k in range(N):
                    matrix[i * N + k, j * N + k] += dSIGS[i][j][k]

    return matrix

def NOISE_dNUFIS_1D_matrix(group, N, chi_p, chi_d, dNUFIS, k_complex, Beff, keff):
    """
    Construct the noise perturbed fission production matrix for 1D geometry.
    
    Parameters:
        group (int): Number of energy groups.
        N (int): Number of spatial cells.
        chi_p (array): Prompt fission spectrum.
        chi_d (array): Delayed fission spectrum.
        dNUFIS (array): Perturbed nu-fission cross sections.
        k_complex (complex): Complex eigenvalue.
        Beff (float): Effective delayed neutron fraction.
        keff (float): Effective multiplication factor.
    
    Returns:
        lil_matrix: Sparse matrix for noise perturbed fission production.
    """
    # Initialize the full matrix with zeros
    matrix = lil_matrix((group*N,group*N), dtype=complex)
    for i in range(group):
        for j in range(group):
            for k in range(N):
                matrix[i*N + k, j*N + k] += (chi_p[i][k] * (1-Beff)/keff + chi_d[i][k] * k_complex) * dNUFIS[j][k]

    return matrix

###################################################################################################
def interpolate_dPHI_rbf_1D(dPHI_zero, group, N, map_detector, rbf_function=None):
    """
    Interpolate dPHI values in 1D using RBF interpolation.

    Parameters:
        dPHI_zero (array): Input complex array with zeroed elements.
        group (int): Number of energy groups.
        N_max (int): Total number of elements in the 1D grid.
        conv (array): Conversion array mapping indices.
        map_detector (array): Binary array indicating detector positions.
        rbf_function (str): RBF kernel function (e.g., 'multiquadric', 'gaussian').

    Returns:
        dPHI_interp_new (array): Interpolated complex array.
    """
    # Reshape the input into a 2D array of shape (group, N_max)
    dPHI_zero_array = np.reshape(np.array(dPHI_zero), (group, N))
    dPHI_interp_array = dPHI_zero_array.copy()

    for g in range(group):
        dPHI_zero_real = np.real(dPHI_zero_array[g])
        dPHI_zero_imag = np.imag(dPHI_zero_array[g])

        # Get non-zero coordinates and values for real and imaginary parts
        coords_real = np.array([n for n in range(N) if map_detector[n] == 1]).reshape(-1, 1)
        values_real = np.array([dPHI_zero_real[n] for n in coords_real.flatten()])
        coords_imag = coords_real  # Same coordinates for real and imaginary parts
        values_imag = np.array([dPHI_zero_imag[n] for n in coords_imag.flatten()])

        # Calculate epsilon based on the pairwise distances
        pairwise_distances = np.diff(coords_real.flatten())
        avg_distance = np.mean(pairwise_distances)
        epsilon = avg_distance / (32 * np.max(values_real))
        
        # Create RBF interpolators
        rbf_real = RBFInterpolator(coords_real, values_real, epsilon=epsilon, kernel=rbf_function)
        rbf_imag = RBFInterpolator(coords_imag, values_imag, epsilon=epsilon, kernel=rbf_function)

        # Interpolate for zero elements
        zero_coords = np.array([n for n in range(N) if map_detector[n] == 0]).reshape(-1, 1)
        interpolated_real = rbf_real(zero_coords)
        interpolated_imag = rbf_imag(zero_coords)

        # Handle NaN values using nearest interpolation
        if np.any(np.isnan(interpolated_real)):
            interpolated_real[np.isnan(interpolated_real)] = griddata(
                coords_real.flatten(), values_real, zero_coords[np.isnan(interpolated_real)], method='nearest'
            )
        if np.any(np.isnan(interpolated_imag)):
            interpolated_imag[np.isnan(interpolated_imag)] = griddata(
                coords_imag.flatten(), values_imag, zero_coords[np.isnan(interpolated_imag)], method='nearest'
            )

        # Assign interpolated values back to the array
        for idx, n in enumerate(zero_coords.flatten()):
            dPHI_interp_array[g, n] = interpolated_real[idx] + 1j * interpolated_imag[idx]

    # Flatten the array back to list
    dPHI_interp = dPHI_interp_array.tolist()

    # Convert back to compact representation if necessary
    if len(dPHI_zero) == group * N:
        dPHI_interp_new = np.zeros((group * N), dtype=complex)
        for g in range(group):
            for n in range(N):
                dPHI_interp_new[g * N + n] = dPHI_interp[g][n]
    else:
        dPHI_interp_new = dPHI_interp

    return dPHI_interp_new