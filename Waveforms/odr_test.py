import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import numpy as np
from scipy.odr import *

def plot_odr(file1,file2, low_cut, high_cut):

    # Find all CSV files that start with "TEK"
    # csv_files = glob.glob("TEK*.csv")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    data1 = pd.read_csv(file1, skiprows=16)
    
    # Assuming the first column is x and the second is y
    x1 = data1.iloc[:, 0]
    y1 = data1.iloc[:, 1]
    
    data2 = pd.read_csv(file2, skiprows=16)
    
    # Assuming the first column is x and the second is y
    x2 = data2.iloc[:, 0]
    y2 = data2.iloc[:, 1]

    d1 = np.array([x1, y1]).transpose()
    d2 = np.array([x2, y2]).transpose()
    d1 = np.array([d for d in d1 if d[0] < high_cut and d[0] > low_cut]).transpose()
    d2 = np.array([d for d in d2 if d[0] < high_cut and d[0] > low_cut]).transpose()   
    
    x1_func = interp1d(x1, y1, fill_value=np.inf, kind="linear", bounds_error=False)
    cut_x1_func = interp1d(d1[0], d1[1], fill_value=np.inf, kind="linear", bounds_error=False)
    x2_func = interp1d(x2, y2, fill_value=np.inf, kind="linear", bounds_error=False)
    cut_x2_func = interp1d(d2[0], d2[1], fill_value=np.inf, kind="linear", bounds_error=False)

    # Define residuals function fitting x2 to x1

    #Definite fitted long path function
    def fit_function(B,x):
        return B[0]*x2_func(x + B[2]) + B[1]
    
    #Create model
    fit_model = Model(fit_function)

    #Create data instance
    sx = 5e-10
    sy = 5e-9
    #fit_data = Data(d1[0], d1[1], wd=1./np.power(sx,2), we=1./np.power(sy,2))
    fit_data = Data(d1[0], d1[1])

    
    #Instantiate regression
    fit_odr = ODR(fit_data, fit_model, beta0=[2, 0, 3e-7], maxit=5000)
    fit_odr.set_iprint(final=1)

    #Run the fit
    fit_output = fit_odr.run()
    fit_output.pprint()

    # def residuals(p):
    #     # params p: a, b, t
    #     return [x1_func(d2[0,i]-p[2]) - (cut_x2_func(d2[0,i])*p[0] + p[1]) for i in range(len(d2[0]))]
    
    # Use curve_fit to determine best a, b, and c
    #result = least_squares(residuals, x0=[0.5, 0, 3e-7], bounds=([0,-0.6,0],[np.inf,0.6,1e-6]))  # Initial guess: a=1, b=0, c=0
    a,b,t = fit_output.beta

    print(fit_output.beta)

    ax1.plot(x1, y1, label="Short Path", color='blue')
    ax1.plot(x2, y2, label="Long Path",  color='orange')

    # Apply transformation using best-fit parameters
    x3 = d2[0]-t
    y3 = d2[1]*a + b

    ax1.plot(x3, y3, label="Fit: $\Delta$t = " + str(round(t*(10**9),2)) + "ns", ls='-', color='red', lw=1)
        

    # Configure the plot
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (V)")
    ax1.set_title("Interferometer Laser Pulse Signals")
    ax1.legend()
    ax1.grid()
    ax1.set_xlim(-1e-5,-0.2e-5)

    #orthonormal_err = np.sqrt((fit_output.delta/)**2 + fit_output.eps**2) * np.sign(fit_output.eps)

    ax2.scatter(d2[0], fit_output.eps,s=0.1, color='red')
    # Configure the plot
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Residual (V)")
    ax2.set_title("Fit Vertical Residuals")
    ax2.grid()
    ax2.set_xlim(low_cut, high_cut)

    ax3.scatter(d2[0], fit_output.delta,s=0.1, color='red')
    # Configure the plot
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Residual (s)")
    ax3.set_title("Fit Horizontal Residuals")
    ax3.grid()
    ax3.set_xlim(low_cut, high_cut)

    plt.tight_layout()
    plt.show()

    # hess_inv = np.linalg.inv(np.dot(result.jac.T, result.jac))
    # mse = (np.array(residuals(result.x))**2).sum()/(len(d2[0])-3)
    # sigma = np.sqrt(np.diag(hess_inv * mse))
    # u_t = sigma[2]

    u_t = fit_output.sd_beta[2]**2
    mse = fit_output.sum_square/(len(d2[0])-3)

    print("t " + str(t))
    print("u_t " + str(u_t))
    print("mse " + str(mse))


    d = 102.572
    u_d = 0.204
    c = d/t
    u_c = c * np.sqrt((u_d/d)**2 + (u_t/t)**2)

    print("c = " + str(c))
    print("u_c = " + str(u_c))
    print("low " + str(low_cut))
    print("high " + str(high_cut))

    print("avg_err_t " + str(np.mean(fit_output.delta)))


plot_odr("short.CSV", "long.CSV", -0.8e-5, -0.4e-5)
# cubic
# t 3.391959957340952e-07
# u_t 3.974849188205679e-10
# mse 4.815591638116638e-06
# c = 302397437.73511094
# u_c = 698055.588094813
# low -8e-06
# high -4e-06



plot_odr("short_zoom.CSV", "long_zoom.CSV", -0.8e-5, -0.4e-5)
# cubic
# t 2.971774696793284e-07
# u_t 7.083870332858979e-06
# mse 0.025530432216563537
# c = 345154025.675907
# u_c = 8227495758.346236
# low -8e-06
# high -4e-06

