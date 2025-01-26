import statistics
import random
import pandas as pd
import plotly.express as px
import numpy as np




"""
think about 3 Dimensions



a dimension is a line


[min,max]

PSO just steps towards global Target with some step size


if we have a value 

D0 [=======X======]
D1 [=======1======]
D2 [=======2======]
here we have full infomation about X with respect to D
but we are evaluing a already known point X in different dimensions (expoitation of D0~D1,2)
if we want to maximise exploration we need to find a point that is furthest from all points (increases infomation)

to maximise explotation we want to take the Global best, zoom in by 10x 
calulate gradients to closest points choose new


exploring have step size
when we exploting have exact value




D0 [=======X======]
D1 [=======1======]
D2 [=======2======]


D0 [=====X=O======]
D1 [=====1=1======]
D2 [=2=====2======]

if dimensions are independant
infomation about a new point X in dimension D gives infomension about ~D

"""






















class particle_swarm():
    """
    Super Simple Particle swarm Optimiser
    Best paritcle extrapolates avg of others
    """
    def __init__(self,n,factor,input_dims,history):
        self.history_size = history
        self.particles_n = n
        self.factor  = factor
        self.score_history = [[] for i in range (n)] # ouput
        self.domain_history = [[] for i in range (n)] # input
        self.last_particle_best =  [0 for i in range (n)]
        self.last_tend_to_global = [0.1 for i in range (n)]
        self.last_step_size = [1 for i in range (n)]
        self.current_particle_i = 0
        self.last_update = [[0 for i in range(n)] for i in range (input_dims)]
        self.best_postitons = [[0 for i in range(n)] for i in range (input_dims)]
        self.bounds = [0,1]
        self.blacklist  =[]
        self.ndims = input_dims
        self.itr = 0

        self.bbest = -9999

    def add_new_postion_score(self,paticle_i,domain,score):
        self.itr +=1
        self.score_history[paticle_i].append(score)
        self.domain_history[paticle_i].append(domain)

        over_history = len(self.score_history[paticle_i]) > self.history_size
        if over_history:
            self.score_history[paticle_i] = self.score_history[paticle_i][-self.history_size:]
            self.domain_history[paticle_i] = self.domain_history[paticle_i][-self.history_size:]
        

    def get_best_input(self,particle_i):
        best_score_per_particle = [max(i) for i in self.score_history]
        best_score = max(best_score_per_particle)

        # update global best
        if best_score> self.bbest:
            self.bbest = best_score
            self.best_score = best_score

            # recalculate domains can be multiple
            best_domains = []
            for parti,part in enumerate(self.score_history):
                    #print(part)
                    if int(max(part)==self.best_score):
                        for histi,p in enumerate(part):
                            #print(p)
                            if p==self.best_score:
                                best_domains.append(self.domain_history[parti][histi])

            self.best_postitons = best_domains
            #print(len(self.best_postitons))

        if len(self.best_postitons) > 1:
            
            current_pos = self.domain_history[particle_i][-1]
            last_dist = 999999
            for pos in self.best_postitons:
                
                dist = 0     
                for di,d in enumerate(pos):
                    dist += abs(current_pos[di]-d) # abs of the 2

                
                if last_dist>dist:
                    #find 
                    best_pos = pos

            #best_pos = self.domain_history[dmin_idx[0]][dmin_idx[1]]
            self.best_domain = best_pos #lowest distance pos

            #print(f'multi target {particle_i} {self.best_domain}')
        else:
            
            self.best_domain = self.best_postitons[0]
            #print(f'single target {particle_i} {self.best_domain}')

        return self.best_domain

    def update_postiton(self,particle_i):
        
        current_pos = self.domain_history[particle_i][-1]
        self.get_best_input(particle_i)
        best_pos = self.best_domain

        # if sum(self.score_history[particle_i])==0 and len(self.score_history)==10 and self.itr>100: 
        #     self.blacklist.append(particle_i)
        #     return best_pos

        
        

        # if best_pos == current_pos: 

        #     # TODO find point with max distance from other points, 
        #     # or next best probe or probe with list of minimum visted quadrents
        #     #print(f'stop {particle_i}')
        #     self.blacklist.append(particle_i)
        #     return current_pos

        new_position = []
        for i,(j,k) in enumerate(zip(current_pos,best_pos)):
            
            exploration_step_size = self.last_step_size[particle_i]

            score_per_particle = [max(_) for _ in self.score_history]
            particle_best = score_per_particle[particle_i]
            max_score_opt = (1+particle_best)/(1+self.best_score)
            tend_to_global = 1#random.random()#(1-max_score_opt)*1 + (max_score_opt)*random.random()#)#

            #factor = min(abs(self.factor * (1-1/max_score_opt)),self.factor)

            
            #tend to radnom best
            rp = int(random.random()*self.particles_n)
            rbest = score_per_particle[rp]
            idx = self.score_history[rp].index(rbest)
            particle_best_domain = self.domain_history[rp][idx]

            # rdirection = (1+particle_best_domain[i] - j)/abs(1+particle_best_domain[i] - j)
            # rdistance = factor*abs(particle_best_domain[i]-j) 


            # # tend to current
            # idx = score_per_particle.index(particle_best)
            # particle_best_domain = self.domain_history[idx][0]


            # # Tend to local Best
            # direction = (1+particle_best_domain[i] - j)/abs(1+particle_best_domain[i] - j)
            # distance = factor*abs(particle_best_domain[i]-j) 

            # if k-j==0: # could be local minima
            #     #we won
            #     #print(f"stop {particle_i}")
            #     self.blacklist.append(particle_i)
            #     return current_pos


            # # Tend to global
            # global_direction = (k - j)/abs(k - j) 
            # global_distance = abs(k-j)

            # if global_distance<2*exploration_step_size: exploration_step_size/=2

            # velocity_to_particle_rbest = rdirection*(factor*rdistance)
            # velocity_to_particle_best = direction*(factor*distance)
            # velocity_to_global_best = global_direction*(factor*global_distance)
            velocity = grad_fn(current_pos)[i]
            if velocity==0:
                #we won
                #print(f"stop {particle_i}")
                self.blacklist.append(particle_i)
                return current_pos

            new_velocity = (j
                            #+ (1-tend_to_global)*velocity_to_particle_rbest
                            #+(1-tend_to_global)*velocity_to_particle_best 
                            + self.factor*velocity
                            #+ (tend_to_global)*velocity_to_global_best
            )
            #print(j,velocity,self.factor,float(self.factor)*velocity)
            #print(grad_fn(current_pos)[i])
            new_pos = j
            #if i == 0:
            new_pos = j+self.factor*velocity

            if len(self.score_history) >= 2:
                if self.score_history[-2][particle_i] > 0 and self.score_history[-1][particle_i] < 0: #asmitope
                    self.factor = 1
                    x0 = (self.score_history[-1][particle_i][0]-self.score_history[-2][particle_i][0])/2
                    x1 = (self.score_history[-1][particle_i][1]-self.score_history[-2][particle_i][1])/2
                    return [x0,x1]

                if self.score_history[-2][particle_i] < 0 and self.score_history[-1][particle_i] > 0: #asmitope
                    self.factor = 1
                    x0 = (self.score_history[-1][particle_i][0]-self.score_history[-2][particle_i][0])/2
                    x1 = (self.score_history[-1][particle_i][1]-self.score_history[-2][particle_i][1])/2
                    return [x0,x1]

            #print(j,new_pos,velocity)
            #input(new_pos)
            new_position.append(new_pos)
            self.last_update[i][particle_i] = new_pos


            self.last_particle_best[particle_i] = particle_best
            self.last_tend_to_global[particle_i] = tend_to_global
            self.last_step_size[particle_i] = exploration_step_size
            
        return new_position
    
    def next_particle(self):

        for i in range(self.particles_n):
            self.current_particle_i = (self.current_particle_i + 1) % self.particles_n #increase by one untile reached available
            if self.current_particle_i not in self.blacklist: break


    def rank_particles(self):
        """
        Particle ranking dynamically choose particles to update
        """
        ranks = []
        for particle_i in range(self.particles_n):
            score_effciencay_rank = self.last_particle_best[particle_i]/(self.best_score+1)
            gradient_rank = 0
            rank = score_effciencay_rank+gradient_rank
            ranks.append(rank)

        return ranks.index(max(ranks))







import math

def fn(x):
    #https://en.wikipedia.org/wiki/Test_functions_for_optimization
    #return math.cos(x[0]/4) + 5*math.sin(x[1]*20) - 2*math.sin(x[1]) + 0.25*math.cos(x[1]-100)
    #return -0.0001*(abs(math.sin(x[0])*math.sin(x[1])**(abs(100-((x[1]**2+x[1]**2)**0.5/math.pi))))+1)**0.1
    a=abs(100-math.sqrt(x[0]**2+x[1]**2)/math.pi)
    b=abs(math.sin(x[0])*math.sin(x[1])*math.exp(a))+1
    c=-0.0001*b**0.1
    return c

def fn(x):
    stem = 0
    for _x in x:
        
        stem+= _x**2-10*math.cos(2*math.pi*_x)

    rg = len(x)*10 + stem + 0.33
    return -rg

# def fn(x):
#     stem = 0
#     for i in range(len(x)-1):
#         stem+= (x[i+1]-x[i]**2)**2 + (1-x[i])**2
        
#     return -stem

# def fn(x):
#     return -(2*x[0]**2-1.05*x[0]**4+x[0]**6/6 + x[0]*x[1] + x[1]**2)



t0 = 75
small_core = 1276
C = 1/15
t_offset = -25

f_offset = 950
f_mult = 1/2

def fn(x):
    vcore = x[0]
    frequency = x[1]
    
    # Stage 2: Minimum Vcore penalty
    vmin = frequency*f_mult + f_offset  # vmin calculation based on frequency
    
    # Stage 3: Temperature penalty
    T = vcore*C + t_offset
    pto = (1+math.e**-0.2*(t0-T))
    penalty_t = 0.99/pto

    pvo = (1+math.e**0.5*(vmin-vcore))
    penalty_v = 0.99/pvo
    
    # Final hashrate
    hashrate = (frequency * small_core * penalty_v * penalty_t)/1000
    #print(x,penalty_v,penalty_t,hashrate)
    return hashrate

    


def noisy_fn(x):
    hashrate = fn(x)
    noisy_hashrate = np.random.normal(hashrate,max(1,hashrate/8))
    return noisy_hashrate



def derivative_vcore(x_v, x_f, k_v, k_t, k_f):
    """
    Compute the derivative with respect to x_v.

    Parameters:
    - x_v: float, the vcore value
    - x_f: float, the frequency value
    - k_v: float, constant related to vcore
    - k_t: float, constant threshold for temperature
    - k_f: float, constant related to frequency

    Returns:
    - derivative: float, the computed derivative
    """
    # Exponential terms
    exp1 = np.exp(0.5 * k_f * x_f - 0.5 * x_v)
    exp2 = np.exp(0.2 * k_v * x_v - 0.2 * k_t)
    
    # Denominator components
    denom1 = (exp1 + 1)**2
    denom2 = (exp2 + 1)**2
    
    # Numerator components
    term1 = 0.5 * exp1 * (exp2 + 1)
    term2 = 0.2 * k_v * (exp1 + 1) * exp2
    
    # Full derivative
    derivative = (x_f * (term1 - term2)) / (denom1 * denom2)
    
    return derivative

def derivative_frequency(x_v, x_f, k_v, k_t, k_f):
    """
    Compute the scaled derivative with respect to x_f.

    Parameters:
    - x_v: float, the vcore value
    - x_f: float, the frequency value
    - k_v: float, constant related to vcore
    - k_t: float, constant threshold for temperature
    - k_f: float, constant related to frequency
    - small_core: float, scaling factor for small core performance

    Returns:
    - derivative: float, the computed scaled derivative
    """
    # Exponential terms
    exp1 = np.exp(0.5 * k_f * x_f)
    exp2 = np.exp(0.5 * x_v)
    exp3 = np.exp(0.2 * k_t)
    exp4 = np.exp(0.2 * k_v * x_v)

    exp1 = min(exp1,1000)
    exp2 = min(exp2,1000)
    exp3 = min(exp3,1000)
    exp4 = min(exp4,1000)
    
    # Numerator components
    term1 = -0.5 * k_f * x_f * exp1
    term2 = exp1
    term3 = exp2
    numerator = (term1 + term2 + term3) * exp3 * exp2
    
    # Denominator components
    denom1 = (exp1 + exp2)**2
    denom2 = (exp3 + exp4)

    
    # Full derivative
    derivative = numerator / (denom1 * denom2)
    
    return derivative

# Gradient function (partial derivatives of hashrate)
def grad_fn(x):
    vcore = x[0]
    frequency = x[1]

    # Stage 2: Minimum Vcore penalty
    vmin = frequency*f_mult + f_offset  # vmin calculation based on frequency
    
    # Stage 3: Temperature penalty
    
    T = vcore*C + t_offset

    pto = (1+math.e**-0.2*(t0-T))
    penalty_t = 0.99/pto

    #penalty_t = 0.99/(1+math.e**-0.2*(t0-vcore*C - 25))

    pvo = (1+math.e**0.5*(vmin-vcore))
    penalty_v = 0.99/pvo

    # pvo = (1+math.e**0.5*(frequency/2 + 950-vcore))
    # ddpenaltyv_v = 0.600465/((vmin-vcore+0.606531)**2)
    # ddpenaltyt_v = 0.810543*C/(0.818731*(-C*vcore+t0 + t_offset))
    # ddpenaltyv_f = -1.63223/(f_mult*(1.64872*(frequency/vcore)+f_offset-vcore)**2)
    # #ddpenaltyt_f = 0.810543*C/(0.818731*(-C*vcore+t0 + t_offset))

    # d_hashrate_d_frequency = small_core/1000 *(penalty_t  * ddpenaltyv_f)
    # d_hashrate_d_vcore = small_core/1000 *(frequency  *  ddpenaltyv_v * ddpenaltyt_v)


    d_hashrate_d_vcore = small_core/1000*derivative_vcore(vcore,frequency,C,t0,f_mult)
    d_hashrate_d_frequency =  small_core/1000*derivative_frequency(vcore,frequency,C,t0,f_mult)
    print(d_hashrate_d_vcore,d_hashrate_d_frequency)

    return [d_hashrate_d_vcore, d_hashrate_d_frequency]


# def fn(x):
#     vcore = x[0]
#     frequency = x[1]

#     vmin_for_f = 1000+frequency/2

#     temp = vcore/15-25


#     #vmin_factor = (vcore-vmin_for_f)/abs(vcore-vmin_for_f)#int(vcore>=vmin_for_f) 

#     vmin_factor =1/(1+math.e**(-0.1*(vcore-vmin_for_f)))

#     too_hot = int(temp>58)

#     sign = 1-(1+(58-temp)/abs(58-temp)) /2 # 0,1

#     temp_p = min(58/temp,1)
#     #hot_penalty = 1-((temp-58)/7)
#     hot_penalty = -1/(1+500000000000*math.e**(-0.4*temp)) + 1

    
#     #print(x,temp,too_hot,hot_penalty,vmin_for_f,vmin_factor)
#     hashrate = hot_penalty*too_hot*vmin_factor*1276*frequency/1000
#     return hashrate


#
#hashrate = 1-((vcore/a-b-58)/7)*(vcore-c+frequency/d)*1276*frequency/1000



use_rank_boosting = False

def run(ite):

    n = 1
    pso = particle_swarm(n,15,2,10)
    data = []
    p_i = 0

    #probes = [[400,1000],[400,1200],[500,1100],[500,1200]]
    # for d1 in range(1100,1350,100):
    #     for d2 in range(400,700,100):
    #         probe = [d1,d2]
    #         y = fn(probe)
    #         pso.add_new_postion_score(p_i%n,probe,y)
    #         data.append([0,0,p_i%n,-2,-2,probe[0],probe[1],fn(probe)])
    #         p_i = (p_i +1) % n


    probes = [[1600,700]]
    for i in range(len(probes)):
        probe = probes[i]
        y = fn(probe)
        pso.add_new_postion_score(i%n,probe,y)
        #pso.get_best_input()
        t = math.floor(i/n)
        best_score_per_particle = 0#[max(i) for i in pso.score_history]
        data.append([0,0,i%n,-2,-2,probe[0],probe[1],y])

    
    probes = 0
    for i in range(probes):
        probe = [random.random()*500+900,300+random.random()*500] #[random.random()*5-2.5,random.random()*5-2.5]# 
        y = fn(probe)
        pso.add_new_postion_score(i%n,probe,y)
        
        t = math.floor(i/n)
        best_score_per_particle = 0#[max(i) for i in pso.score_history]
        data.append([0,0,i%n,-2,-2,probe[0],probe[1],y])
    
    for i in range(4000):
        new_position = pso.update_postiton(pso.current_particle_i)
        y = fn(new_position)
        pso.add_new_postion_score(pso.current_particle_i,new_position,y)
        t = math.floor((i+probes)/n)
        best_score_per_particle = [max(i) for i in pso.score_history]
        data.append([t,i,pso.current_particle_i,pso.best_score,best_score_per_particle[pso.current_particle_i],new_position[0],new_position[1],y])
        if int(i/50) % 2 and use_rank_boosting:
            pso.current_particle_i = pso.rank_particles()
        else:
            pso.next_particle()

        if set(pso.blacklist) == set(list(range(pso.particles_n))): break
        
        
        #print(i,pso.current_particle_i,pso.best_domain,pso.best_score)


    frame =pd.DataFrame(data,columns="t,i,p_i,best_global,best_local,x,y,score".split(","))
    frame.loc[:,'convergence'] = frame.best_local/frame.t
    #print(frame)
    #print(frame.best_global.describe())
    if ite==0:
        px.scatter(frame,x='x',y='y',color='score').show()
        px.scatter(frame,x='t',y='convergence',color='p_i').show()
    return frame.best_global.max(),frame.loc[frame.best_global==frame.best_global.max()].t.min()
    


bests = []
bests2 = []
for i in range(1): 
    best_val,t_end = run(i)
    bests.append(float(best_val))
    bests2.append(t_end)

print(statistics.mean(bests))
print(statistics.mean(bests2))
