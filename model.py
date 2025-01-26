import math
import random
import traceback
import statistics
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

e_approx = 2.71828


def solve_vmin_p(X,Y):
    """
    Solve for Vmin and p using given (Vcore, Pv) data.

    Parameters:
        data (list of tuples): Each tuple contains (Vcore, Pv) values.

    Returns:
        tuple: (p, Vmin)
    """
    # Perform least squares regression
    n = len(X)
    sum_X = sum(X)
    sum_Y = sum(Y)
    sum_XY = sum(X[i] * Y[i] for i in range(n))
    sum_X_squared = sum(X[i] ** 2 for i in range(n))
    print(n *sum_X_squared , sum_X ** 2 )
    # Calculate slope (-p) and intercept (b)
    slope = (n * sum_XY - sum_X * sum_Y) / (n * sum_X_squared - sum_X ** 2)
    intercept = (sum_Y - slope * sum_X) / n
    
    # Calculate p and Vmin
    p = -slope
    Vmin = intercept / p
    
    return p, Vmin

def linear_regression(X,Y):
    """
    Solve for k and c using linear regression.

    Parameters:
        data (list of tuples): Each tuple contains (F, Vcore, Pv).

    Returns:
        tuple: (k, c)
    """

    n = len(X)
    sum_X = sum(X)
    sum_Y = sum(Y)
    sum_XY = sum(X[i] * Y[i] for i in range(n))
    sum_X_squared = sum(X[i] ** 2 for i in range(n))
    if (n * sum_X_squared == sum_X ** 2):
        return 0, statistics.median(Y)
    
    # Calculate slope (-p) and intercept (b)
    slope = (n * sum_XY - sum_X * sum_Y) / (n * sum_X_squared - sum_X ** 2)
    intercept = (sum_Y - slope * sum_X) / n
    
    
    return slope, intercept


def solve_linear_least_squares(x, y):
    """
    Solve Y = mx + c using the least squares method for given x and y data points.

    Parameters:
        x (list): List of x values.
        y (list): List of y values.

    Returns:
        tuple: Slope (m) and intercept (c) of the best-fit line.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("The lengths of x and y must be equal.")

    # Compute sums needed for the formulas
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x_squared = sum(x[i] ** 2 for i in range(n))

    # Calculate slope (m) and intercept (c)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    c = (sum_y - m * sum_x) / n

    return m, c


def fn(x):
    small_cores = 1276
    vcore = x[0]
    frequency = x[1]
    
    # Stage 2: Minimum Vcore penalty
    vmin = frequency*0.333 + 950  # vmin calculation based on frequency
    
    # Stage 3: Temperature penalty
    T = vcore/15 -25
    pto = (1+e_approx**(-0.02*(100-T)))
    penalty_t = 1/pto

    pvo = (1+e_approx**(0.01*(vmin-vcore)))
    penalty_v = 1/pvo
    
    # Final hashrate
    Hashrate = small_cores/1000 * (frequency * penalty_v *penalty_t)
    #print('fn',x,penalty_v,penalty_t,hashrate)
    return Hashrate,T,vcore,penalty_v, penalty_t

def noisy_fn(x):
    data = fn(x)
    noisy_hashrate = max(1e-7,np.random.normal(data[0],data[0]/5))#5
    return noisy_hashrate,*data[1:]

def linear_regression2(T,y):
    """
    Perform linear regression to find p and t0 from y = p * (t0 - T).
    
    Parameters:
        data (list of tuples): Each tuple contains (T, Pt), where T is the temperature,
                               and Pt is the probability.
    
    Returns:
        tuple: (p, t0)
    """
    # Calculate y = ln(1 / Pt - 1) and store T values
    
    # Compute averages
    n = len(y)
    avg_y = sum(y) / n
    avg_T = sum(T) / n
    
    # Compute required sums
    sum_T_y = sum(T * y for T, y in zip(T, y))
    sum_T_squared = sum(T ** 2 for T in T)
    
    # Linear regression formulas
    p = (sum_T_y - n * avg_T * avg_y) / (sum_T_squared - n * avg_T ** 2)
    t0 = avg_y / p + avg_T
    
    return p, t0

class parametrized_model():
    def __init__(self,bounds,small_cores):
        self.history = []
        self.bounds = bounds
        self.step = 0
        self.small_cores = small_cores
        self.is_trained = False

    def add_point(self,x,H,t,vcore_actual):
        self.history.append([x,H,t,vcore_actual])

    def reparametrize_T_eqn(self,history):
        # Calulate T (SH) (vcore changes)
        vcores = [h[0][0] for h in history]
        ts = [h[2] for h in history]
        m,c = solve_linear_least_squares(vcores,ts)
        self.k_tmult = m
        self.k_toffset = c

    def reparametrize_vmin_eqn(self,history,with_t=False,use_p=None):
        #Calulate Vmin (freq changes only)
        expected_hashrates =  [(h[0][1]*self.small_cores/1000) for h in history]
        if with_t:
            expected_hashrates =  [(h[0][1]*self.small_cores/1000)*(1/(1+math.e**(self.tpower*(self.t0-h[2])))) for h in history]
        vcores = [h[0][0] for h in history]
        freqs  = [h[0][1] for h in history]

        Vp = [h[1]/exp for h,exp in zip(history,expected_hashrates)]
        p = 1
        if use_p is not None:
            p = use_p
        self.vpower = p
        vmins = []
        for vp,h in zip(Vp,history):
            vcore = h[0][0]
            pbest_err = 10000
            bests = []
            for vmin in range(900,1350):
                err_per_v = 0
                Vp_remake = 1/(1+math.e**(p*(vmin-vcore)))#[1/(1+math.e**(p*(h[0][0]-vmin))) for h in history]
                err = abs(vp-Vp_remake)#sum([abs(y-tp1) for y,tp1 in zip(Vp,Vp_remake)])
                err_per_v+=err
                if err <= pbest_err:
                    pbest_err = err
                    bests.append([vmin,p,err])

                #if err_per_v <= pbest_err:
                #    pbest_err = err_per_v

   
            best_min_val = min([i[2] for i in bests])
            bests = [i for i in bests if i[2]==best_min_val]
            vmax = max([i[0] for i in bests])
            vmin = min([i[0] for i in bests])
            #print(vmin,vmax)
            vmin_idx = [i[0] for i in bests].index(vmax)
            best_values = bests[vmin_idx]
            self.vpower = best_values[1]
            vmins.append(best_values[0]) # p does change best

        k,c = linear_regression(freqs,vmins)
        self.k_fvmult = k
        self.k_fvoffset = c

    def reparametrize_vpower(self,history):
        err_min = 100000000
        vpower_val = 0
        for vpower in range(1,100):
            vpower = vpower/100
            self.reparametrize_vmin_eqn(history,True,vpower)
            err = self.history_error()
            if err < err_min:
                err_min = err
                vpower_val = vpower
        
        self.reparametrize_vmin_eqn(history,True,vpower_val)
        #print("best vpower",vpower_val)



    def reparametrize_t0_eqn(self,history):
        #Calc T0 (vcore changes) we need penalty V to be 1
        #ln(exp/h-1) = p*(t0-T)
        vmins = [h[0][1]*self.k_fvmult + self.k_fvoffset for h in history]
        vps = [1-(1/(1+math.e**(self.vpower*(h[0][0]-vm)))) for h,vm in zip(history,vmins)]   

        expected_hashrates =  [(h[0][1]*self.small_cores/1000)*vp for h,vp in zip(history,vps)] # only T penalty left

        Tp = [h[1]/eh for h,eh in zip(history,expected_hashrates)]
        lhs = [(1/tp-0.5) for tp in Tp] # becasue its not -1, a multiple is required
        ys = [4*math.log(r,math.e) for r in lhs]

        #search for min

        best_err = 10000
        pbest_err = 10000
        tbest = 0
        bbest = 0
        for t0 in range(75,90):
            err_per_deg = 0
            for j in range(50):
                p = (-j-1)/100

                Tp_remake = [1/(1+math.e**(p*(t0-h[2]))) for h in history]
                err = sum([abs(y-tp1) for y,tp1 in zip(Tp,Tp_remake)])
                err_per_deg+=err
                
                if err <= pbest_err:
                    pbest_err = err
                    bbest = t0
                    #
                    #self.t0 = t0
            #err = sum([(y-tp1)**2 for y,tp1 in zip(Tp,Tp_remake)])
            if err_per_deg <= best_err:
                best_err = err
                tbest = t0

        self.t0 = tbest#statistics.mean([bbest,tbest])

        p = [y/(self.t0-h[2]) for h,y in zip(history,ys)]
        self.tpower = statistics.mean(p)

        #print("new t0",self.tpower,self.t0,bbest,tbest)



    def model(self,x):
        vcore = x[0] 
        frequency = x[1]

        # Calulate T
        T = vcore*self.k_tmult + self.k_toffset

        #Calulate Vmin
        vmin = frequency*self.k_fvmult + self.k_fvoffset

        penalty_v = 1/(1+e_approx**(self.vpower*(vmin-vcore)))
        penalty_t = 1
        #penalty_t = 1/(1+e_approx**(self.tpower*(self.t0-T)))
        
        if self.vpower==0:
            penalty_v = 1
        if self.tpower==0:
            penalty_t = 1

        
        
        Hashrate = self.small_cores/1000 * (frequency  * penalty_t * penalty_v)
        return Hashrate,T,penalty_v,penalty_t

    def get_best(self):
        hashrate_max = 0
        best_pos = []
        for h in self.history:
            if h[1] > hashrate_max:
                best_pos = h[0]
                hashrate_max = h[1]

        real_hashrate = fn(best_pos)
        return best_pos,hashrate_max,real_hashrate

    def maximise_eqn(self):
        """
        Query function
        """
        max_y_est = 0
        best_pos = []
        for vcore in range(self.bounds[0][0],self.bounds[0][1],10):
            for frequency in range(self.bounds[1][0],self.bounds[1][1],10):
                y_est = self.model([vcore,frequency])[0]
                if y_est>max_y_est:
                    best_pos = [vcore,frequency]
                    max_y_est = y_est
        
        return best_pos
    
    def maximise_fn(self):
        max_y_est = 0
        best_pos = []
        for vcore in range(self.bounds[0][0],self.bounds[0][1],10):
            for frequency in range(self.bounds[1][0],self.bounds[1][1],10):
                y_est = fn([vcore,frequency])[0]
                if y_est>max_y_est:
                    best_pos = [vcore,frequency]
                    max_y_est = y_est
        return best_pos,max_y_est
    

    def history_error(self):
        total_err = 0
        for h in self.history:
            err = abs(h[1]-self.model(h[0])[0])
            total_err += err

        return total_err/len(self.history)
    

    def reparametrize_model_hashrate_err(self,n,data):

        #optimise history with respect to Hashrate error
        min_err = 1000000
        best_hist = []
        best_T_hist = []
        for _j,j in enumerate(self.history[:n]):
            for i in self.history:
                history = [i,j]
                if _j==0:
                    vcore_difference = i[0][0]!=j[0][0]
                    if vcore_difference: 
                        try:
                            self.reparametrize_T_eqn(history)
                            best_T_hist = [i,j]
                        except ZeroDivisionError:
                            # this occours in reparametrize T
                            #print(i[0],j[0],"Z")
                            pass
                        except OverflowError:
                            #print(i[0],j[0],"O")
                            pass
                        except ValueError:
                            #print(i[0],j[0],"V")
                            pass

                if _j > 0:
                    if best_T_hist:
                        self.reparametrize_T_eqn(best_T_hist)
                    else: 
                        break
                    
                    try:    
                        self.reparametrize_vmin_eqn(history)
                        self.reparametrize_t0_eqn(history)
                        self.reparametrize_vpower(history)
                        hash_rate_err= 0
                        
                        for core,freq,temp,hashrate in data:
                            ypred = self.model([core,freq])
                            errH = abs(hashrate-ypred[0])
                            errT = abs(temp-ypred[1])
                            hash_rate_err+=errH+errT

                        if hash_rate_err<min_err:
                            min_err = hash_rate_err
                            best_hist = [i,j]

                        
                        #print(i[0],j[0],hash_rate_err)
                    except ZeroDivisionError:
                        # this occours in reparametrize T
                        #print(i[0],j[0],"Z")
                        pass
                    except OverflowError:
                        #print(i[0],j[0],"O")
                        pass
                    except ValueError:
                        #print(i[0],j[0],"V")
                        pass
          
        if best_hist and best_T_hist: 
            try:
                self.reparametrize_T_eqn(best_T_hist)
                self.reparametrize_vmin_eqn(best_hist)
                self.reparametrize_t0_eqn(best_hist)
                self.reparametrize_vpower(best_hist)
                self.is_trained = True
            except Exception as e:
                print("model fail ",best_hist)
                print(traceback.format_exc())
        else:
            print('no best hist')

    def reparametrize_model_all(self):
        try:
            self.reparametrize_T_eqn(self.history)
            self.reparametrize_vmin_eqn(self.history)
            self.reparametrize_t0_eqn(self.history)
            self.reparametrize_vpower(self.history)
        except Exception as e:
                print(traceback.format_exc())


def sigmoid(x):
    res = 0
    try:
        res = 1/(1+math.e**(-x))
    except OverflowError:
        pass
    return res

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def eval_penalty(a,b,p):
    return sigmoid(p*(a-b))
    
ERR_MAX = 999999999
class parametrized_model2():
    def __init__(self,small_cores):
        self.vmin_lbound = 1000
        self.vmin_hbound = 1400
        self.vpower_lbound = 1
        self.vpower_hbound = 500
        self.fpower_lbound = -500
        self.fpower_hbound = -1
        self.fmax_lbound = 500
        self.fmax_hbound = 800
        self.small_cores = small_cores
        self.history = []
        self.chips =1
        self.hint_counter = 0

    def add_point(self,x,H,t,eff):
        if H >150:
            self.history.append([x,H,t,eff])



    def reparm_vmin(self,history,linear_reg_max,use_f=False):
        #calulate vmin
        y_vmin = []
        x_fs = []
        errs =[]
        for h in history:
            V = h[0][0]
            F = h[0][1]
            
            best_err = ERR_MAX
            best_vmin = -1
            
            for vmin in range(self.vmin_lbound,self.vmin_hbound):
                
                hashrate_est = self.eval_Expected_H(h[0])*eval_penalty(V,vmin,self.vmin_power)
                if use_f:
                    fmax = F*self.fmax_term0 + V* self.fmax_term1 + self.fmax_const
                    hashrate_est =self.eval_Expected_H(h[0])*eval_penalty(V,vmin,self.vmin_power)*eval_penalty(F,fmax,self.fpower)
                err = abs(h[1]-hashrate_est)
                
                if err<best_err:
                    
                    best_vmin = vmin
                    best_err = err

            if best_vmin != -1:
                x_fs.append(F)
                y_vmin.append(best_vmin)
                errs.append(best_err)
        
        if len(x_fs) > 3:
            #print(x_fs,y_vmin)
            sz = len(x_fs)
            x_fs = [x for _,x in sorted(zip(errs,x_fs))][:min(sz,linear_reg_max)]
            y_vmin = [x for _,x in sorted(zip(errs,y_vmin))][:min(sz,linear_reg_max)]
            
            m,c = linear_regression(x_fs,y_vmin)
            self.vmin_mult_term = m
            self.vmin_const_term = c 
            #print("new vmin terms",self.vmin_mult_term,self.vmin_const_term,self.calc_err(history))



            all_good = True
        else:
            all_good = False

        return all_good,self.calc_err(self.history)


    def reparam_fmax(self,history,linear_reg_max):
        y_fmax = []
        x_fs = []
        x_vs = []
        errs = []
            
            
        for h in history:
            best_err = ERR_MAX
            best_fmax = -1
            V = h[0][0]
            F = h[0][1]
            for fmax in range(self.fmax_lbound,self.fmax_hbound):
                
                vmin = F*self.vmin_mult_term+self.vmin_const_term
                est_1 = self.eval_Expected_H(h[0])*eval_penalty(V,vmin,self.vmin_power)
                hashrate_est = est_1*eval_penalty(F,fmax,self.fpower)
                #print(V,F,fmax,est_1,hashrate_est,h[1])
                err = abs(h[1]-hashrate_est)
                
                if err<best_err:
                    
                    best_err = err
                    best_fmax = fmax

            if best_fmax != -1:
                #print(F,V,best_fmax,best_err)
                x_fs.append(F)
                x_vs.append(V)
                y_fmax.append(best_fmax)
                errs.append(best_err)
        
        if len(x_fs) > 3:
            sz = len(x_fs)
            x_fs = [x for _,x in sorted(zip(errs,x_fs))][:min(sz,linear_reg_max)]
            x_vs = [x for _,x in sorted(zip(errs,x_vs))][:min(sz,linear_reg_max)]
            y_fmax = [x for _,x in sorted(zip(errs,y_fmax))][:min(sz,linear_reg_max)]

            fm,cf = linear_regression(x_fs,y_fmax)
            vm,cv = linear_regression(x_vs,y_fmax)

            xs = [0.5*fm*f+0.5*vm*v for f,v in zip(x_fs,x_vs)]
            m,c = linear_regression(xs,y_fmax)
            
            self.fmax_term0 = m*fm*0.5
            self.fmax_term1 = m*vm*0.5
            self.fmax_const = c
            #print([[f for f,v in zip(x_fs,x_vs)]])
            #print([[fm*f+cf for f,v in zip(x_fs,x_vs)]])
            #print([[vm*v+cv for f,v in zip(x_fs,x_vs)]])
            # print([[m*0.5*fm*f+m*0.5*vm*v+c for f,v in zip(x_fs,x_vs)]])
            #print(y_fmax)
            # print(self.calc_err(self.history))
            
            #print("new f term",self.fmax_term0,self.fmax_term1,self.fmax_const,self.calc_err(self.history))

            
            all_good = True
        else:
            all_good = False

        return all_good,self.calc_err(self.history)


    def reparam_vpower(self,history,use_f):

        best_err = ERR_MAX
        best_vpower = -1
        
        for vpower in range(self.vpower_lbound,self.vpower_hbound):
            vpower/=1000
            err = 0
            for h in history:
                V = h[0][0]
                F = h[0][1]
                vmin = F*self.vmin_mult_term+self.vmin_const_term
                hashrate_est =self.eval_Expected_H(h[0])*eval_penalty(V,vmin,vpower)
                if use_f:
                    fmax = F*self.fmax_term0 + V* self.fmax_term1 + self.fmax_const
                    hashrate_est =self.eval_Expected_H(h[0])*eval_penalty(V,vmin,self.vmin_power)*eval_penalty(F,fmax,self.fpower)
                err += abs(h[1]-hashrate_est)

            if err<best_err:
                best_vpower = vpower
                best_err = err

        if best_vpower != -1:
            self.vmin_power = best_vpower
            #print("new vpower",self.vmin_power,self.calc_err(self.history))
            all_good = True
        else:
            all_good = False
        return all_good,self.calc_err(self.history)


    def reparam_fpower(self,history):
        best_err = ERR_MAX
        best_fpower = -1
        
        for fpower in range(self.fpower_lbound,self.fpower_hbound):
            fpower /= 1000
            err = 0
            for h in history:
                V = h[0][0]
                F = h[0][1]
                vmin = F*self.vmin_mult_term + self.vmin_const_term
                fmax = F*self.fmax_term0 + V* self.fmax_term1 + self.fmax_const
                hashrate_est =self.eval_Expected_H(h[0])*eval_penalty(V,vmin,self.vmin_power)*eval_penalty(F,fmax,fpower)
                err += abs(h[1]-hashrate_est)

            if err<best_err:
                best_fpower = fpower
                best_err = err

        if best_fpower != -1:
            self.fpower = best_fpower
            #print("new fpower",best_fpower,self.calc_err(self.history))
            all_good = True
        else:
            all_good = False
        return all_good,self.calc_err(self.history)
    



    def reparam_err(self,history):

        xs = []
        x_fs = []
        x_vs = []
        ys = []
        for h in history:
            V = h[0][0]
            F = h[0][1]
            vmin = F*self.vmin_mult_term+self.vmin_const_term
            fmax = F*self.fmax_term0 + V* self.fmax_term1 + self.fmax_const
            hashrate_est = self.eval_Expected_H(h[0])*eval_penalty(V,vmin,self.vmin_power)*eval_penalty(F,fmax,self.fpower)
            xs.append(hashrate_est)
            ys.append(h[1])
            x_fs.append(F)
            x_vs.append(V)
        
        if len(xs) > 3:
            sz = len(xs)
            m,c = linear_regression(xs,ys)
            if m < 0:
                m = 1
                c = 0
            # hashratye_est*m + c = hashrate

            self.err_m =  m
            self.err_c =  c
            
            all_good = True
        else:
            all_good = False

        return all_good,self.calc_err(self.history)

    def assumption_detect(self):
        "inverse correlation detection freq should always equal"
        xs = []
        xfs = []
        xvs = []
        ys = []
        for h in self.history:
            xs.append(h[0][1]*h[0][0])
            xfs.append(h[0][1])
            xvs.append(h[0][0])
            ys.append(h[1])

        
        m,c = linear_regression(xs,ys)
        mf,c = linear_regression(xfs,ys)
        mv,c = linear_regression(xvs,ys)
        assumptions =  1> m >0 and -1 < mf < 0 and mv > 1.5
        return assumptions


            
    def build_partial(self,history): 
        all_good = True
        sorted_n = 5
        best_data_points = []
        current_best_err = self.calc_err(self.history)
        last_point = self.history[-1]

        for i,h in enumerate(history[:-1]):
            h2 = last_point
            j = len(history)-1
            if h!=h2:
                h3 = history[(i+len(history)//2) % len(history)]
                h4 = history[(j+len(history)//2) % len(history)]
                tmp = [h,h2,h3,h4]
                all_good,best_score = self.reparm_vmin(tmp,sorted_n,False)
                if all_good:
                    all_good,best_score = self.reparam_vpower(tmp,False)
                if all_good:
                    all_good,best_score = self.reparam_err(tmp)

                assumption = self.assumption_detect()
                if best_score< current_best_err and all_good and assumption:
                    current_best_err = best_score
                    best_data_points = tmp

        if best_data_points != []:
            self.best_data_points = best_data_points

        all_good,best_score = self.reparm_vmin(self.best_data_points,sorted_n,False)
        if all_good:
            all_good,best_score = self.reparam_vpower(self.best_data_points,False)
        if all_good:
            all_good,best_score = self.reparam_err(tmp)
 

        if all_good:
            self.is_trained = True
        else:
            self.is_trained = False




    def build(self,history):
        #Sanity test
        self.vmin_power = 1
        self.fpower = 1
        self.vmin_const_term = 10
        self.fmax_const = 10
        self.fmax_term0 = 0
        self.fmax_term1 = 0
        self.vmin_mult_term =0
        self.err_c = 0
        self.err_m = 1

        all_good = True

        sorted_n = 5
        best_match = 1000
        best_data_points = []


        #deterministic again
        for i,h in enumerate(history):
            for j,h2 in enumerate(history):
                if h!=h2:
                    h3 = history[(i+len(history)//2) % len(history)]
                    h4 = history[(j+len(history)//2) % len(history)]
                    tmp = [h,h2,h3,h4]
                    all_good,best_score = self.reparm_vmin(tmp,sorted_n,False)
                    if all_good:
                        all_good,best_score = self.reparam_vpower(tmp,False)
                    if all_good:
                        all_good,best_score = self.reparam_err(tmp)

                    assumption = self.assumption_detect()
                    if best_score< best_match and all_good and assumption:
                        best_match = best_score
                        best_data_points = tmp

        self.best_data_points = best_data_points


        all_good,best_score = self.reparm_vmin(best_data_points,sorted_n,False)
        if all_good:
            all_good,best_score = self.reparam_vpower(best_data_points,False)
        if all_good:
            all_good,best_score = self.reparam_err(tmp)
 

        if all_good:
            self.is_trained = True
        else:
            self.is_trained = False
            #raise ValueError("not trained")

        
    def get_history_best(self):
        hashrate_max = 0
        best_pos = []
        for h in self.history:
            if h[1] > hashrate_max:
                best_pos = h[0]
                hashrate_max = h[1]
        return best_pos,hashrate_max



    def maximise_hashrate_eqn(self,bounds,hint=False):
        pos = []
        scores_per_freq = []
        best = 0


        attempts = []
        for frequency in range(bounds[1][0],bounds[1][1],10):
            for vcore in range(bounds[0][0],bounds[0][1],10):
                pos = [vcore,frequency]
                if pos not in [h[0] for h in self.history]:
                    attempts.append([vcore,frequency])

        scores_per_freq = {}
        for vcore,frequency in attempts:
            if scores_per_freq.get(frequency,None) is None:
                scores_per_freq.update({frequency:[]})

            scores_per_freq[frequency].append(self.evaluate([vcore,frequency]))

        for key in scores_per_freq:
            filtered = sorted([f for f in scores_per_freq[key]])[-3:] #top3
            mean = 0
            if len(filtered)>1:
                mean = statistics.mean(filtered)
            scores_per_freq.update({key:mean})

        best_freq = max(scores_per_freq, key=scores_per_freq.get)
        
        err = self.calc_err(self.history)
        best_pos = -1
        if err<200:
            for pos in attempts:
             
                    eval = self.evaluate(pos)
                    if eval > best:
                        best = eval
                        best_pos = pos
        else:
            best_poss = []
            for pos in attempts:
                if pos[1]==best_freq:
                    best_poss.append(pos)

            if best_poss:
                idx =int(len(best_poss)*random.random())
                best_pos = best_poss[idx]



                

        if best_pos ==-1:
            # entropy source
                idx =int(len(attempts)*random.random())
                best_pos = attempts[idx]


        # if not hint:

            
        # else:
        #         self.hint_counter +=1
        #         opts = [0.125,0.375,0.625,0.875]

        #         idx = int(opts[self.hint_counter%len(opts)]*len(attempts))
        #         best_pos = attempts[idx]

        

        
        # #detect statics 
        # if len(self.history) > 2:
        #     static_v = statistics.stdev([h[0][0] for h in self.history[-3:]])
        #     static_f =  statistics.stdev([h[0][1] for h in self.history[-3:]])
        #     hashrate_m,hc = linear_regression([0,1,2],[h[1] for h in self.history[-3:]])
        #     mv,hc = linear_regression([0,1,2],[h[0][0] for h in self.history[-3:]])
        #     mf,hc = linear_regression([0,1,2],[h[0][1] for h in self.history[-3:]])
        #     print(int(static_f),int(static_v),mv,mf,hashrate_m)

        return best_pos

    

    def eval_Expected_H(self,x):
        F = x[1]
        expectation_hashrate = self.small_cores*self.chips*F
        return expectation_hashrate/1000



    def evaluate(self,x):
        F = x[1]
        V = x[0]
        
        vmin = (F*self.vmin_mult_term+self.vmin_const_term)
        penalty_vmin = eval_penalty(V,vmin,self.vmin_power)

        fmax = F*self.fmax_term0 + V* self.fmax_term1 + self.fmax_const
        penalty_f = eval_penalty(F,fmax,self.fpower)
        #print(penalty_f,penalty_vmin)

        hashrate_est = self.eval_Expected_H(x)*penalty_vmin*penalty_f

        hashrate_est = hashrate_est*self.err_m + self.err_c
        return hashrate_est
        

    def calc_err(self,history):
        total_err = 0
        for h in history:
            # MAE
            err = abs(h[1]-self.evaluate(h[0]))
            total_err += err

        if len(history) == 0:
            history = [0]
        return total_err/len(history)





def optimize():
    OPT = parametrized_model([[1000,1350],[300,700]],1276)
    # 3 configurations to get model
    probes = [[1150,490],[1300,490],[1150,550],[1150,300]]
    n = 100
    hashrates  = []
    for p in  [probes[0]]*n:
        hashrate,T,vcore,pv,pt = noisy_fn(p)
        hashrates.append(hashrate)
    OPT.add_point(p,statistics.mean(hashrates),T,vcore)    
    hashrates  = []
    for p in  [probes[1]]*n:
        hashrate,T,vcore,pv,pt = noisy_fn(p)
        hashrates.append(hashrate)
    OPT.add_point(p,statistics.mean(hashrates),T,vcore)
        

    OPT.reparametrize_T_eqn(OPT.history) # first 2



    hashrates  = []
    for p in  [probes[2]]*n:
        hashrate,T,vcore,pv,pt = noisy_fn(p)
        hashrates.append(hashrate)
    OPT.add_point(p,statistics.mean(hashrates),T,vcore) 

    history = [OPT.history[-1],OPT.history[0]]
    OPT.reparametrize_vmin_eqn(history) #probe 2,0


    history = [OPT.history[-1],OPT.history[0]]
    OPT.reparametrize_t0_eqn(history)  #probe 2,0

    hashrates  = []
    for p in  [probes[3]]*n:
        hashrate,T,vcore,pv,pt = noisy_fn(p)
        hashrates.append(hashrate)
    OPT.add_point(p,statistics.mean(hashrates),T,vcore) 
        
    
    history = [OPT.history[3],OPT.history[1]]  #probe 3,0
    OPT.reparametrize_vpower(history)
    
    #history = [OPT.history[3],OPT.history[1]]
    #OPT.reparametrize_vpower(history)
    #OPT.reparametrize_vmin_eqn(history,True,OPT.vpower)


    input("donw")
    #history = [OPT.history[-2],OPT.history[0]]
    #OPT.reparametrize_t0_eqn(history)


    # we have learned we have the model
    suggestion = OPT.maximise_eqn()
    y_pred = OPT.model(suggestion)
    hashrate,T,vcore,pv,pt = noisy_fn(suggestion)
    OPT.add_point(suggestion,hashrate,T,vcore)

    #show the heatmap of the learned model
        
    print('Vmin',OPT.vpower,OPT.k_fvmult,OPT.k_fvoffset)
    print('T',OPT.tpower,OPT.k_tmult,OPT.k_toffset)
    print('T0',OPT.t0)

    print(OPT.maximise_fn())
    print(OPT.get_best())
    input("graph")

    # Parameter grid
    vcore_values = np.linspace(1000, 1400, 50)  # Example range for Vcore
    f_values = np.linspace(400, 800, 50)      # Example range for F
    grid = [(v, f) for v in vcore_values for f in f_values]

    # Query the model and compute errors
    predicted_scores = [OPT.model(x)[0] for x in grid]
    true_scores = [fn(x)[0] for x in grid]
    errors = [abs(p - t) for p, t in zip(predicted_scores, true_scores)]

    # Reshape for visualization
    vcore_grid, f_grid = np.meshgrid(vcore_values, f_values)
    predicted_grid = np.array(predicted_scores).reshape(len(f_values), len(vcore_values))
    error_grid = np.array(errors).reshape(len(f_values), len(vcore_values))
    true_grid = np.array(true_scores).reshape(len(f_values), len(vcore_values))

    # Visualization
    plt.figure(figsize=(18, 6))

    # Plot 1: Predicted Scores
    plt.subplot(1, 3, 1)
    plt.contourf(vcore_grid, f_grid, predicted_grid, cmap='viridis')
    plt.colorbar(label='Predicted Score')
    plt.title('Predicted Scores')
    plt.xlabel('Vcore')
    plt.ylabel('F')

    # Plot 2: True Scores
    plt.subplot(1, 3, 2)
    plt.contourf(vcore_grid, f_grid, true_grid, cmap='viridis')
    plt.colorbar(label='True Score')
    plt.title('True Scores')
    plt.xlabel('Vcore')
    plt.ylabel('F')

    # Plot 3: Error Heatmap
    plt.subplot(1, 3, 3)
    plt.contourf(vcore_grid, f_grid, error_grid, cmap='plasma')
    plt.colorbar(label='Error')
    plt.title('Error Heatmap')
    plt.xlabel('Vcore')
    plt.ylabel('F')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    optimize()
