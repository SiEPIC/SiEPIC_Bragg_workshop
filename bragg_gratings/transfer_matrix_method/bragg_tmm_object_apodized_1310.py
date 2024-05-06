# -*- coding: utf-8 -*-
"""
@author: Mustafa Hammood
@adapted by: Bobby Zou to add apodization
"""
# %%


class bragg_wg:
    def __init__(
        self,
        width=0.35e-6,
        thickness=220e-9,
        dw=40e-9,
        period=270e-9,
        N=300,
        alpha_dBcm=3,
        wavl_start=1260e-9,
        wavl_stop=1360e-9,
        resolution=0.1e-9,
        gaussianIndex=2
    ):
        self.width = width
        self.thickness = thickness
        self.dw = dw
        self.period = period
        self.N = N
        self.alpha_dBcm = alpha_dBcm
        self.wavl_start = wavl_start
        self.wavl_stop = wavl_stop
        self.resolution = resolution
        self.gaussianIndex = gaussianIndex

        self.poly, self.n1_reg, self.n2_reg, self.n3_reg = self.neff_lookup(
            width, thickness
        )
        self.vis_shown = False

        self.neff0_values = [
            self.neff0(wavl, width, thickness) for wavl in self.lambda_0
        ]

        # automated class initialization, breaks debugger..?
        # params = locals()
        # for name, value in params.items():
        #     if name != 'self':
        #         setattr(self, name, value)

    @property
    def lambda_0(self):
        import numpy as np

        return np.linspace(
            self.wavl_start,
            self.wavl_stop,
            round((self.wavl_stop - self.wavl_start) / self.resolution),
        )

    @property
    def l(self):
        return self.period / 2

    @property
    def kappa(self):
        # TODO: load a lookup table instead...
        # Waveguide dimension 500nm x 220nm

        # return -1.53519e19 * self.dw**2 + 2.2751e12 * self.dw
        # Adjusted to waveguide dimension 350nm x 220nm

        # period = 265 nm
        # return -8.90545e19 * self.dw ** 2 + 9.77165e12 * self.dw - 5.66549e3
        
        # period = 270 nm
        return -6.62634e19 * self.dw ** 2 + 8.73236e12 * self.dw + 6.27178e3
        
        # period = 275 nm
        # return -6.11280e19 * self.dw ** 2 + 8.57437e12 * self.dw + 9.04911e3


    @property
    def lambda_bragg(self):
        # TODO: calculate phase match condition instead...
        return 1305.72e-9

    @property
    def n_delta(self):
        return self.kappa * self.lambda_bragg / 2

    def n_delta_param(self, dwidth):
        
        # period = 265 nm
        # kappa = -8.90545e19 * dwidth ** 2 + 9.77165e12 * dwidth - 5.66549e3
        
        # period = 270 nm
        kappa = -6.62634e19 * dwidth ** 2 + 8.73236e12 * dwidth + 6.27178e3
        
        # period = 275 nm
        # kappa = -6.11280e19 * dwidth ** 2 + 8.57437e12 * dwidth + 9.04911e3
        
        return kappa * self.lambda_bragg / 2

    @property
    def alpha(self):
        import numpy as np

        return np.log(10) * self.alpha_dBcm / 10 * 100.0  # per meter

    def euclidean_distance(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def normalize(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    def neff_lookup(self, width, thickness):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        import numpy as np

        # load waveguide neff data from lookup table
        file_path = "wg_data/wg_variability_1310.txt"
        column_names = [
            "thickness",
            "width",
            "wavl_start",
            "wavl_stop",
            "n1",
            "n2",
            "n3",
        ]

        data = np.loadtxt(file_path, delimiter=",")
        points = data[:, :2]  # width and thickness
        n_values = data[:, 4:]

        # Create polynomial features for the input points
        poly = PolynomialFeatures(degree=2)
        points_poly = poly.fit_transform(points)

        # Create a polynomial fit for n1, n2, and n3
        n1_reg = LinearRegression().fit(points_poly, n_values[:, 0])
        n2_reg = LinearRegression().fit(points_poly, n_values[:, 1])
        n3_reg = LinearRegression().fit(points_poly, n_values[:, 2])

        return poly, n1_reg, n2_reg, n3_reg

    def neff0(self, lambda0, width, thickness, visualize=True):
        import numpy as np
        import matplotlib.pyplot as plt

        point = np.array([[thickness, width]])
        point_poly = self.poly.transform(point)

        neff1 = self.n1_reg.predict(point_poly)[0]
        neff2 = self.n2_reg.predict(point_poly)[0]
        neff3 = self.n3_reg.predict(point_poly)[0]

        neff0_value = neff1 + neff2 * lambda0 + neff3 * lambda0**2

        if visualize and not self.vis_shown:
            width_range = np.linspace(self.width - 30e-9, self.width + 30e-9, 100)
            thickness_range = np.linspace(
                self.thickness - 20e-9, self.thickness + 20e-9, 100
            )

            width_grid, thickness_grid = np.meshgrid(width_range, thickness_range)
            neff0_grid = np.zeros_like(width_grid)

            for i in range(len(thickness_range)):
                for j in range(len(width_range)):
                    point = np.array([[thickness_range[i], width_range[j]]])
                    point_poly = self.poly.transform(point)

                    neff1 = self.n1_reg.predict(point_poly)[0]
                    neff2 = self.n2_reg.predict(point_poly)[0]
                    neff3 = self.n3_reg.predict(point_poly)[0]

                    neff0_grid[i, j] = neff1 + neff2 * lambda0 + neff3 * lambda0**2

            fig, ax = plt.subplots()
            contour = ax.contourf(
                width_grid * 1e9, thickness_grid * 1e9, neff0_grid, 20, cmap="viridis"
            )
            fig.colorbar(contour)
            ax.set_xlabel("Width (nm)")
            ax.set_ylabel("Thickness (nm)")
            ax.set_title(f"Effective index (neff0) at lambda0 = {lambda0 * 1e9:.2f} nm")
            plt.show()
            self.vis_shown = True

        return neff0_value

    def n1(self, lambda0, width, thickness, index):
        return self.neff0_values[index] - self.n_delta / 2
    
    def n1_param(self, lambda0, dwidth, thickness, index):
        return self.neff0_values[index] - self.n_delta_param(dwidth) / 2

    def n2(self, lambda0, width, thickness, index):
        return self.neff0_values[index] + self.n_delta / 2
    
    def n2_param(self, lambda0, dwidth, thickness, index):
        return self.neff0_values[index] + self.n_delta_param(dwidth) / 2

    def HomoWG_Matrix(self, wavl, neff, l):
        import math
        import cmath
        import numpy as np

        j = cmath.sqrt(-1)
        beta = 2 * math.pi * neff / wavl - j * self.alpha / 2
        v = [np.exp(j * beta * l), np.exp(-j * beta * l)]
        T_hw = np.diag(v)
        return T_hw

    def IndexStep_Matrix(self, neff1, neff2):
        import numpy as np

        a = (neff1 + neff2) / (2 * np.sqrt(neff1 * neff2))
        b = (neff1 - neff2) / (2 * np.sqrt(neff1 * neff2))

        T_is = [[a, b], [b, a]]
        return T_is

    def optimized_matrix_mult(self, T):
        from functools import reduce
        import numpy as np

        matrices = T[:]
        result = []

        while len(matrices) > 1:
            temp = []
            for i in range(0, len(matrices), 2):
                if i + 1 < len(matrices):
                    temp.append(np.dot(matrices[i], matrices[i + 1]))
                else:
                    temp.append(matrices[i])
            matrices = temp
        return matrices[0]

    def Grating_Matrix(self, wavl, l, index):
        import numpy as np
        import random
        import math

        T = []
        i = 0
        while i < self.N:
            
            if self.gaussianIndex == 0:
                dwidth_apodized = self.dw
            else:
                # apodization paramater addon
                profileFunction = math.exp(-0.5*(2*self.gaussianIndex*(i-self.N/2)/(self.N))**2)
                
                # From KLayout PCell "amf_bragg_apodized.py"
                # "profile = int(round(self.corrugation_width/2/dbu))*profileFunction"
                # Breaking down into setps from the above equation, I had to mimic the implmentation in KLayout but it is not necessary
                profile = self.dw/2/1e-9
                profile = int(round(profile))
                profile = profile * profileFunction

                # Get total dwidth
                # Option 1 - No apodization
                # dwidth_apodized = self.dw
                
                # Option 2 - With apodization
                dwidth_apodized = (profile * 2 * 1e-9)
            
            # Calculate the total width of the bragg for sanity check
            total_width_1 = (self.width + dwidth_apodized)*1e6
            total_width_2 = (self.width - dwidth_apodized)*1e6

            n1_final = self.n1_param(wavl, dwidth_apodized, self.thickness, index)
            n2_final = self.n2_param(wavl, dwidth_apodized, self.thickness, index)
            l_final = l + 0

            T_hw1 = self.HomoWG_Matrix(wavl, n1_final, l_final)
            T_is12 = self.IndexStep_Matrix(n1_final, n2_final)
            T_hw2 = self.HomoWG_Matrix(wavl, n2_final, l_final)
            T_is21 = self.IndexStep_Matrix(n2_final, n1_final)
            
            Tp1 = np.matmul(T_hw1, T_is12)
            Tp2 = np.matmul(T_hw2, T_is21)
            Tp = np.matmul(Tp1, Tp2)
            T.append(Tp)
            i += 1
        if self.N == 1:
            return T[0]
        else:
            return self.optimized_matrix_mult(T)

    def Grating_RT(self, wavl, index):
        import numpy as np

        M = self.Grating_Matrix(wavl, self.l, index)
        T = np.absolute(1 / M[0][0]) ** 2
        R = np.absolute(M[1][0] / M[0][0]) ** 2.0  # or M[0][1]?
        return [T, R]

    def Run(self):
        self.R = []
        self.T = []
        self.R, self.T = zip(
            *[self.Grating_RT(wavl, index) for index, wavl in enumerate(self.lambda_0)]
        )
        import scipy.io

        data_save = {
            'R': self.R,
            'T': self.T,
            'lambda_0': self.lambda_0
        }
        # Save to a .mat file
        scipy.io.savemat('braggResponse.mat', data_save)

    def visualize(self):
        import matplotlib.pyplot as plt
        import numpy as np

        if "self.R" not in globals() or "self.T" not in globals():
            self.Run()
            print("Simulation data not found, running simulation...")

        fig, ax = plt.subplots()
        ax.plot(
            self.lambda_0 * 1e9,
            10 * np.log10(self.T),
            label="Reflection",
            color="blue",
        )
        ax.plot(
            self.lambda_0 * 1e9, 10 * np.log10(self.R), label="Transmission", color="red"
        )
        ax.set_ylabel("Response (dB)", color="black")
        ax.set_xlabel("Wavelength (nm)", color="black")
        ax.set_title("Calculated response of the structure using TMM (dB scale)")
        ax.legend()


if __name__ == "__main__":
    bragg = bragg_wg(
        period=270e-9, dw=10e-9, N=500, width=350e-9, thickness=220e-9, gaussianIndex=2
    )
    bragg.visualize()

# %%