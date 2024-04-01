from scipy.optimize import broyden1
import numpy as np
from matplotlib import pyplot as plt


c1 = 1.38
c2 = 8.05
c3 = 0.0252
c4 = 0.487
a = 0.922
Rec = 3.401
n = 4

def solve(Pr, Ra):
    def f(x):
        f_val = (1 + x**n)**(-1/n)
        return f_val

    def g(x):
        g_val = x * (1 + x**n)**(-1/n)
        return g_val
    def guess_regions(Ra, Pr): # from ref- 8
        Re_guess= 84*(Pr**(-0.62))
        Nu_guess= 4.8*(Pr**(0.19))
        if (Ra==10**5 and Pr>=0.001 and Pr<=0.7):
            Re_guess= 84*(Pr**(-0.62))
            Nu_guess= 4.8*(Pr**(0.19))
        elif (Ra==10**6 and Pr>=0.01 and Pr<=0.7):
            Re_guess= 240*(Pr**(-0.65))
            Nu_guess= 9*(Pr**(0.18))
        elif (Ra==10**7):
            if (Pr==0.01):
                Re_guess= 84*(Ra**(0.53))
                Nu_guess= 0.043*(Ra**(0.29))
            elif (Pr==0.7):
                Re_guess= 0.34*(Ra**(0.49))
                Nu_guess= 0.17*(Ra**(0.28))
            elif (Pr==7):
                Re_guess= 0.021*(Ra**(0.54))
                Nu_guess= 0.14*(Ra**(0.30))
        return Re_guess, Nu_guess
    
    def equations(vars):
        Nu, Re = vars
        comp_1 = (c1*Re**3) + (c2*((Re*Re)/g((Rec**0.5/Re**0.5))))
        comp_2 = c3*Pr*Re*f(((2*a*Nu)/(Rec**0.5))*g((Rec/Re)**(0.5)))
        comp_3 = (c4*(Re*Pr)**0.5)*f(((2*Nu*a)/(Rec**(0.5)))*g((Rec/Re)**(0.5)))**(0.5)
        eq1 = (Nu - 1)*Ra*(Pr**(-2)) - comp_1
        eq2 = Nu - comp_2 - comp_3
        return [eq1, eq2]

    # Initial guess for Nu and Re (changed for better convergence)
    Re_guess, Nu_guess= guess_regions(Ra, Pr) 
    initial_guess = [Nu_guess, Re_guess]  # Initial guess for Nu and Re

    # Maximum number of iterations
    max_iter = 100

    # Tolerance for convergence
    tolerance = 1e-6

    try:
        solution = broyden1(equations, initial_guess, f_tol=tolerance, maxiter=max_iter)
        return solution
    except Exception as e:
        # print("Error:", e)
        return initial_guess

def input(Pr, Ra): # For checking with single inputs
    [Nu, Re]= solve(Pr, Ra)
    print(f'Nu: {Nu}, Re: {Re}')
    
def plot(): # function for plotting 
    Ra_ls= [10**5, 10**6]
    Pr_ls= np.linspace(10**(-3), 0.7, 10)
    # Ra_ls= [10**6, 10**7, 10**8, 10**9, 10**10, 10**11, 10**12, 10**13, 10**14, 10**15]
    for Ra in Ra_ls:
        Nu_ls= []
        for Pr in Pr_ls:
            [Nu, Re]=  solve(Pr, Ra)
            Nu_ls.append(Nu)
        plt.plot(Pr_ls, Nu_ls, label= f'For Ra: {Ra}')
    plt.xlabel('lg Pr')
    plt.ylabel('lg Nu')
    plt.xlim(10**(-3), 0.7)
    plt.ylim(0, 20)
    plt.legend()
    plt.show()
plot()
# input(0.7, 1e8)

