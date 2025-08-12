import numpy as np
import json
import time
import os
import sys
from scipy.linalg import lstsq
import scipy.linalg
from itertools import combinations, islice
from math import comb

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

start_time = time.time()

from UTILS import Utils
from MATRIX_BUILDER import *
from METHODS import *
from POSTPROCESS import PostProcessor
from SOLVERFACTORY import SolverFactory

class NoiseGreen:
    def __init__(self, dim, geom_type, PHI, keff, group, N, BC, D, TOT, SIGS, chi, NUFIS, precond, v, Beff, omega, l, dTOT, dSIGS, dNUFIS, solver_type, output_dir, case_name, dSOURCE=None, dx=None, dy=None, dz=None, h=None, x=None, I_max=None, J_max=None, K_max=None):
        self.dim = dim
        self.geom_type = geom_type
        self.solver_type = solver_type
        self.output_dir = output_dir
        self.case_name = case_name
        self.PHI = PHI
        self.keff = keff
        self.group = group
        self.N = N
        self.x = x
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dSOURCE = dSOURCE
        self.h = h
        self.BC = BC
        self.D = D
        self.TOT = TOT
        self.SIGS = SIGS
        self.chi = chi
        self.NUFIS = NUFIS
        self.precond = precond
        self.v = v
        self.Beff = Beff
        self.omega = omega
        self.l = l
        self.dTOT = dTOT
        self.dSIGS = dSIGS
        self.dNUFIS = dNUFIS  

    def build_green_function(self):
        if self.geom_type == '1D':
            return self.main_unfold_1D_green()
        elif self.geom_type == '2D rectangular':
            return self.main_unfold_2D_rect_green()
        elif self.geom_type == '2D triangular':
            return self.main_unfold_2D_hexx_green()
        elif self.geom_type == '3D rectangular':
            return self.main_unfold_3D_rect_green()
        elif self.geom_type == '3D triangular':
            return self.main_unfold_3D_hexx_green()
        else:
            raise ValueError("Invalid dimension specified. Only 1, 2, or 3 are allowed.")

    def main_unfold_1D_green(self):
        os.makedirs(f'{self.output_dir}/{self.case_name}_01_GENERATE', exist_ok=True)
        matrix_builder = MatrixBuilderNoise1D(self.group, self.N, self.TOT, self.SIGS, self.BC, self.dx, self.D, self.chi, self.NUFIS, self.keff, self.v, self.Beff, self.omega, self.l, self.dTOT, self.dSIGS, self.dNUFIS)
        M, dS = matrix_builder.build_noise_matrices()
        G_sol = np.ones(self.group*self.N, dtype=complex)
        G_matrix = np.zeros((self.group * self.N, self.group * self.N), dtype=complex)
        if self.precond == 1:
            print('Solving using ILU')
            M_csc = M.tocsc()
            ilu = spilu(M_csc)
            M_preconditioner = LinearOperator(M_csc.shape, matvec=ilu.solve)
        elif self.precond == 2:
            print('Solving using LU Decomposition')
            M_csc = M.tocsc()
            lu = splu(M_csc)
            M_preconditioner = LinearOperator(M_csc.shape, matvec=lu.solve)
        else:
            print('Solving using Solver')
        for g in range(self.group):
            for n in range(self.N):
                dS = [0] * (self.group * self.N)
                m = g*self.N+n
                dS[m] = 1
                errdPHI = 1
                tol = 1E-8
                iter = 0
                while errdPHI > tol:
                    G_solold = np.copy(G_sol)
                    if self.precond == 0:
                        G_sol = spsolve(M, dS)
                    elif self.precond == 1:
                        # Solve the linear system with CG and ILU preconditioning
                        G_sol, info = cg(M, dS, tol=1e-8, maxiter=1000, M=M_preconditioner)
                    errdPHI = np.max(np.abs(G_sol - G_solold) / (np.abs(G_sol) + 1E-20))
                G_sol_reshape = np.reshape(G_sol, (self.group, self.N))
                G_matrix[:, m] = G_sol.flatten()  # Assign solution to row
                # OUTPUT
                output = {}
                for j in range(self.group):
                    G_sol_groupname = f'G{g+1}{j+1}'
                    G_sol_list = [{"real": x.real, "imaginary": x.imag} for x in G_sol_reshape[j]]
                    output[G_sol_groupname] = G_sol_list
                # Save data to JSON file
                with open(f'{output_dir}/{case_name}_01_GENERATE/Green_g{g+1}_n{n+1}.json', 'w') as json_file:
                    json.dump(output, json_file, indent=4)
                print(f'Generated Green Function for group = {g+1}, N = {n+1}')

        return G_matrix

def main_unfold_2D_rect_green(self):
    conv = convert_index_2D_rect(self.D, self.I_max, self.J_max)
    conv_array = np.array(conv)
    max_conv = max(conv)

    os.makedirs(f'{self.output_dir}/{self.case_name}_01_GENERATE', exist_ok=True)
    matrix_builder = MatrixBuilderNoise2DRect(self.group, self.N, conv, self.TOT, self.SIGS_reshaped, self.BC, self.dx, self.dy, self.D, self.chi, self.NUFIS, self.keff, self.v, self.Beff, self.omega, self.l, self.dTOT, self.dSIGS_reshaped, self.dNUFIS)
    M, dS = matrix_builder.build_noise_matrices()

    M_petsc = PETSc.Mat().createAIJ(size=M.shape, csr=(M.indptr, M.indices, M.data), comm=PETSc.COMM_WORLD)
    M_petsc.assemble()

    # PETSc Solver (KSP) and Preconditioner (PC)
    ksp = PETSc.KSP().create()
    ksp.setOperators(M_petsc)
    ksp.setType(PETSc.KSP.Type.GMRES)

    pc = ksp.getPC()
    if self.precond == 0:
        print(f'Solving using Sparse Solver')
        pc.setType(PETSc.PC.Type.NONE)
    elif self.precond == 1:
        print(f'Solving using ILU')
        pc.setType(PETSc.PC.Type.ILU)
        print(f'ILU Preconditioner Done')
    elif self.precond == 2:
        print('Solving using LU Decomposition')
        pc.setType(PETSc.PC.Type.LU)
        print(f'LU Preconditioner Done')

    # Solver tolerances
    ksp.setTolerances(rtol=1e-10, max_it=5000)

    G_sol_all = np.ones(self.group*self.N, dtype=complex)
    G_sol_temp = np.ones(self.group*max_conv, dtype=complex)
    G_matrix = np.zeros((self.group * max_conv, self.group * max_conv), dtype=complex)
    
    for g in range(self.group):
        for n in range(self.N):
            if conv[n] != 0:
                dS = [0] * (self.group * max_conv)
                dS[g*max_conv+(conv[n]-1)] = 1  # Set the relevant entry to 1
                dS_petsc = PETSc.Vec().createWithArray(dS)
                dS_petsc.assemble()

                errdPHI = 1
                tol = 1E-10
                iter = 0

                while errdPHI > tol:
                    G_sol_tempold = np.copy(G_sol_temp)
                    G_sol_temp_petsc = PETSc.Vec().createWithArray(G_sol_temp)

                    # Solve the linear system using PETSc KSP
                    ksp.solve(dS_petsc, G_sol_temp_petsc)

                    # Get result back into NumPy array
                    G_sol_temp = G_sol_temp_petsc.getArray()

                    errdPHI = np.max(np.abs(G_sol_temp - G_sol_tempold) / (np.abs(G_sol_temp) + 1E-20))

                for gp in range(self.group):
                    for m in range(self.N):
                        G_sol_all[gp * self.N + m] = G_sol_temp[gp * max_conv + (conv[m] - 1)]
                G_sol_reshape = np.reshape(G_sol_all, (self.group, self.N))
                G_matrix[:, g*max_conv+(conv[n]-1)] = G_sol_temp.flatten()  # Assign solution to row
                
                # OUTPUT
                output = {}
                for gp in range(self.group):
                    G_sol_groupname = f'G{g+1}{gp+1}'
                    G_sol_list = [{"real": x.real, "imaginary": x.imag} for x in G_sol_reshape[gp]]
                    output[G_sol_groupname] = G_sol_list

                i = n % self.I_max
                j = n // self.I_max

               # Save output to HDF5 file
                hdf5_filename = f'{self.output_dir}/{self.case_name}_01_GENERATE/Green_g{g+1}_j{j+1}_i{i+1}.h5'
                save_output_hdf5(hdf5_filename, output)
                print(f'Generated Green Function for group = {g + 1}, J = {j+1}, I = {i+1}')

    return G_matrix
