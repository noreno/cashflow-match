
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from scipy.optimize import linprog
from datetime import date, timedelta

# Example of cashflow matching
# The objective is to minimize the price of a bond portfolio using either
# - linear programming (LP) allowing for fractional bond positions
# - mixed-integer linear programming (MILP) restricting bond positions to be discrete

# Data: bond prices obtained from FRED and simulated liabilities
# Variables:
# T 	number of liabilities
# n 	number of instruments (bonds)
# l_t 	liability at time t, i = 1, 2, .., T
# p_i 	price of bond i, i = 1, 2, ..., n
# cr_i 	coupon rate of bond i
# mt_i 	maturity time of bond i
# f_ti	cash flow of bond i at time t
# Note: all bonds have face value 100 and all coupon rates are annual

# Decision variables:
# x_i	amount held in bond i (at time 0)
# s_t	surplus held at time t (note s_0 = 0)
# Number of decision variables: n + T

# Optimization problem with constraints:
# Minimize total bond price = sum_i x_i p_i = p'x
# Fx + Rs >= L
# x >= 0, s >= 0
# F: T x n matrix of cash flows
# R: T x T matrix where R_tt = -1, R_t+1,t = 1, and R_ts = 0 otherwise

# Set up data frame for bond payoff among other things
def bondStruct(F, N, c, bt):
    ex = np.ones(N)
    if bt == 'T':
        ex = 0 * ex
        ex[N - 1] = F
    elif bt == 'S':
        ex = (F / N) * ex
    if bt in ['T', 'S']:
        od = F * np.ones(N)
        od[1:] = od[1:] - np.cumsum(ex)[:-1]
        ir = c * od
        po = ex + ir
    elif bt == 'A':
        po = F * c / (1 - (1 + c) ** (-N)) * np.ones(N)
        od = F * np.ones(N)
        ir = np.zeros(N)
        for k in np.arange(N):
            ir[k] = c * od[k]
            ex[k] = po[k] - ir[k]
            if k < N - 1:
                od[k + 1] = od[k] - ex[k]
    return pd.DataFrame(data={'Period': np.arange(N) + 1, 'Debt': od, 'Interest': ir, 'Extraction': ex, 'Payoff': po})


# Example

# Liabilities start at l0 and grow exponentially with g% annually for T years
T = 10
l0 = 15
g = 0.05
l = l0 * (1 + g) ** np.arange(T)
start = date(date.today().year, 1, 1)
end = date.today()

# Fetch bond yields
tickers = ['DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10']
y = pdr.DataReader(tickers, 'fred', end - timedelta(weeks=1), end) / 100
y = y.to_numpy()[-1]
n = len(y)
fv = 100 * np.ones(n, dtype=int)
mt = np.array([1, 2, 3, 5, 7, 10])
cr = 0.03 * np.ones(n, dtype=int) + 0.0025 * (mt - 1)
bt = ['T'] * n
for i in np.arange(n):
    bs = bondStruct(fv[i], mt[i], cr[i], bt[i])
    bs = bs.reindex(range(T), fill_value=0)
    if i == 0:
        F = bs['Payoff'].to_numpy()
    else:
        F = np.c_[F, bs['Payoff'].to_numpy()]

tf = np.arange(T) + 1
p = np.zeros(n)
for i in np.arange(n):
    p[i] = np.sum(F[:,i] * (1 + y[i]) ** (-tf))
s = np.zeros(T)

# Check if calculated yield is equal to market rates
#Fext = np.r_[[-p], F]
#tfext = np.arange(T + 1) + 1
#yhat = np.zeros(n)
#for i in np.arange(n):
#    yhat[i] = irr(Fext[:,i], tfext)

R = np.diff(np.eye(T+1))[:-1,:]

# Specify loss function and inequalities in matrix form
c = np.concatenate((p, s))
A = np.concatenate((F, R), axis=1)
b = l
f = lambda x : np.dot(c, x)  # Cost function
g = lambda x : np.sum(np.sum(xs * A, axis=1) - l)  # Sum of cash flows

opt = 'lp'
if opt.lower() == 'milp':
	# MILP: x_ij must be integer or zero, s_ij can be continuous
	integrality = np.concatenate((3 * np.ones_like(p, dtype=int), 2 * np.ones_like(s, dtype=int)))
else:
	# LP
	integrality = None

# Solve optimization problem
out = linprog(c, A_ub=-A, b_ub=-b, integrality=integrality)
xs = out.x
x = xs[:n]
s = xs[n:]
print(out.message)
if out.success:
    print('x = ', np.round(x, 5))
    print('s = ', np.round(s, 5))
    print('f(x, s) = ', np.round(f(xs), 3), ' (', np.round(out.fun, 3), ')')
    print('g(x, s) = ', g(xs))
