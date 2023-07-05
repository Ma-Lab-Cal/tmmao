"""Optional interface between physics packages and agnostic_invDes. See class docs for more info.

"""


import numpy as np
from copy import deepcopy
from random import uniform, choice
from agnostic_invDes import agnostic_invDes
from agnostic_analysis import agnostic_analysis
from datetime import datetime
from random import random

#Serves as an (optional) intermediary between a physics package and agnostic_invDes. Runs simulations, constructs lists of transfer matrices, their derivatives,
#and cost function derivatives in the format demanded by agnostic_invDes, and interfaces results of an optimization with agnostic_analysis.
#
#Here's how optimization is supposed to work. First, instantiate this object. Next, call set_physics() and set_optimizerFunctions() to gather the physics and optimizer functions, respectively. Next, call set_structure()
#to set initial values for the the optimziation, as well as the type of simulation and any sub/superstrates. Then simply call run_optimization() to run the optimization. run_optimizaiton() will interface with
#agnostic_invDes, calling agnostic_invDes.optimize() multiple times depending upon your discretization map preferences. Results are extracted via the evolution dictionary (self.evo). Each key of self.evo is a "grand iteration",
#a single step of the discretization map. These entries are lists (one element per scipy.optimize iteration) of dictionaries, dictionary i is {'costFunction': value of the cost function AFTER iteration i,'paramters':a concatenated list of the
#                                                                                                                                               x and y parameters AFTER iteration i (this is true even if one of x or y was held fixed)}
#self.evo will be automatically saved for later retrieval, but the attribute persists after optzimization if you wish to immediately perform analysis.
class agnostic_director(agnostic_analysis):
    """Directs an inverse design using agnostic invDes.
    
    agnostic_director is provided as a convenience to the user. It is an optional intermediary between the physics packages developed by myself and agnostic_invDes. It turns your requested initial structure into optParams, makes sure that the
    transfer matrices and derivatives produced by the physics packages end up in the right order and the right format for agnostic_director, and, through the inherited agnostic_analysis class, enables visualization of the results.
    agnostic director is designed to be physics agnostic, so it should be easy to add your own custom physics packages
    
    """
    def __init__(self):
        self.evo={}
        self.all_fields=[]
        self.all_tms=[]
        self.all_dTdphis=[]
        self.all_dLdes=[]
        self.all_dLdphis=[]
        self.all_global_bcs=[]
        self.all_param0s=[]
        self.all_paramNs=[]
        self.all_mat1params=[]
        self.all_mat2params=[]
        self.all_custom_input=[]
        self.all_cf_factors=[]
        self.all_scheduler_factors=[]
        self.all_tracked_tm_info=[]
        self.all_tracked_cf_info=[]
        self.xParamCoords=[]
        self.numpy_zv=[]
        self.all_nonzero_dTdphi=[]
        self.sim_params={'physics':'','third_var':'','simPoint':'','callPoint':''}
        self.indicator_dict={}
        self.debug_verbosity=False
        return

    #Physics package is a class with no __init__ input and with the following methods:
    #tm(x,xp1,y,yp1,mat1params,mat2params,sim_params,tracked_info)=a transfer matrix, a dictionary of tracked_info
    #   tracked_info is a dictionary of anything you want to pass on to the next call of tm, for example wavevector angle in the current layer. tracked_info is reset to an empty dictionary at the start of each new sweep through the layers
    #dtm(x,y,yp1,mat1params,mat2params,sim_params,tracked_info,tm)=The derivative of tm wrt x, the derivative of tm wrt y, the derivative of tm wrt yp1
    #global_bcs(mat1params,mat2params,param0,paramN,sim_params)=(A,B,C), tuple of global boundary matrices
    #costFunc(simPoints,callPoints,third_vars,cur_x,cur_y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input)=value of cost function, some object of additional attributes to be delivered to the derivative functions if desired
    #d_costFunc_fields(simPoints,callPoints,third_vars,cur_x,cur_y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input,costFunc_outputs,chosen_simPoints)=dL/de for every e (one set of e's per simulation point/third variable)
    #d_costFunc_phi(simPoints,callPoints,third_vars,cur_x,cur_y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input,costFunc_outputs,chosen_simPoints)=dL/dpi (just one vector), wrt both x and y. This code will sort out which are actually optParams
    #get_name()=string, the name of the type of physics being used. Will be included as a tag in the evo dictionary preamble, in autogenerated output files, and in the sim_params dictionary.
    #interface_tm(x,y,yp1,mat1Left,mat2Left,mat1Right,mat2Right,sim_params,tracked_info)=a transfer matrix, a dictionary of tracked_info. For interfaces between layers obeying different material libraries, primarily between substrates/superstrates
    #left_tm(yp1,matLeft,mat1Right,mat2Right,sim_params)=a transfer matrix, a dictionary of tracked_info. For first interface with semiinfinite incidence medium
    #right_tm(x,y,mat1Left,mat2Left,matRight,sim_params,tracked_info)=a transfer matrix. For last interface with semiinfinite transmission medium
    #d_interface_tm(x,y,yp1,mat1Left,mat2Left,mat1Right,mat2Right,sim_params,tracked_info,tm)=The derivative of tm wrt x, the derivative of tm wrt y, the derivative of tm wrt yp1
    #d_left_tm(yp1,matLeft,mat1Right,mat2Right,sim_params,tm)=The derivative of tm wrt yp1
    #d_right_tm(x,y,mat1Left,mat2Left,matRight,sim_params,tracked_info)=The derivative of tm wrt x, the derivative of tm wrt y
    #
    #mat1Call,mat2Call,param0Call,paramNCall are callables, taking sim_params dictionary as input and returning a dictionary of material parameters. This dictionary will be sent to matrix builders/cost function. sim_params has keys:
    #   sim_params={'callPoint': simulation point in SI units,'simPoint':simulation point in (possibly scaled) simualtion units,'third_vars':list of third variables,'simType':type of simulation point ('frequency','wavelength', or 'energy'),'physics':type of physics, name given by physics package}
    #CustomInputCall is a callable, taking sim_params dicitonary as input and returning some object to be sent to costFunc. For example, this could be a custom incident acoustic power spectrum which the cost function will utilize.
    def set_physics(self,physics_package,mat1Call,mat2Call,param0Call,paramNCall,customInputCall=''):
        self.pp=physics_package()
        self.mat1Call=mat1Call
        self.mat2Call=mat2Call
        self.param0Call=param0Call
        self.paramNCall=paramNCall
        self.physics=self.pp.get_name()
        self.sim_params['physics']=self.physics
        if callable(customInputCall):
            self.customInputCall=customInputCall
        else:
            def default_customInputCall(sim_params):
                return {}
            self.customInputCall=default_customInputCall
        return

    #cfFactorCall is a callable, taking sim_params dictionary as input and returning a dictionary to be sent to costFunc, containing the weights of each non-regularization term in the cost function sum. Updated every simPoint/third_var combination
    #schedulerCall is a callable, taking it_params dictionary as input and returning a dictionary to be sent to costFunc, containing the weights of each regularization term in the cost function sum. Updated every scipy.minimize() call.
    #dynamic_scheduling is a boolean; True means schedulerCall will, instead of operating as above, take the previous L(float), current L(float), number of iterations(int), current x-values(array-like), current y-values(array-like), 
    #   and current scheduler factors (array-like) as input, and return new scheduler factors (array-like)
    #paramPrunerCall is a callable, taking current optParams (array-like) and cost function (float) as input, and returning new optParams (array-like); OR is a non-callable, in which case external manipulation of optParams is disabled
    def set_optimizerFunctions(self,cfFactorCall,schedulerCall,dynamic_scheduling=False,paramPrunerCall=''):
        self.cfFactorCall=cfFactorCall
        self.schedulerCall=schedulerCall
        self.paramPrunerCall=paramPrunerCall
        self.dynamic_scheduling=dynamic_scheduling
        return
    
    #Wrapper if doing dynamic scheduling. This interfaces with AID
    def scheduler(self,Lprev,L,iterations,x,no_cb=False):
        if self.var_x and self.var_y:
            cur_x=x[:int((len(x)+int(self.simType=='fixedPoint'))/2)]
            cur_y=x[int((len(x)+int(self.simType=='fixedPoint'))/2):]
        elif self.var_x:
            cur_x=x
            cur_y=self.cur_y
        elif self.var_y:
            cur_y=x
            cur_x=self.cur_x
        if no_cb:
            term,self.all_scheduler_factors=self.schedulerCall(Lprev,L,iterations,cur_x,cur_y,self.all_scheduler_factors,no_cb)
            return term
        else:
            self.all_scheduler_factors=self.schedulerCall(Lprev,L,iterations,cur_x,cur_y,self.all_scheduler_factors,no_cb)
            return

    #simType: string, either "independent", in which the optimization/simulation parameters are the thickness x and some other physical parameter y (reractive index, ratio, etc.) of each layer; or "fixedPoint", in which the opt/simParams are the y-values of fixed points and 
    #         the x values are lengths of intervals between each fixed point. The points are linearly interpolated and then broken into num_layers steps within each interval for simulation.
    #num_layers: integer, total number of layers if simType is "independent", or the number of layers per interval if simType is "fixedPoint"
    #initial_layer_thickness: Float or array-like. If float, is the thickness in scaled meters of each layer if simType="independent" or of each interval if simType="fixedPoint". Only the initial value; can change if thickness allowed to be an optimization parameter. 
    #                         If array-like, is the  thickness of each individual layer (simType='independent') or interval (simType='fixedPoint'), which may be variable. Will superseed num_layers/num_intervals if there is a length mis-match
    #initial_ys: array-like, initial values for optimization parameters. If empty, will be autogenerated via uniform distribution within y_bounds. If nonempty, will superseed all other parameters. For example, if len(initial_ys)>num_intervals and simType='fixedPoint',
    #            then more intervals of length equal to the last interval will be added. If simType='independent' and len(initial_ys)<len(initial_layer_thickness), then intial_layer_thickness will be truncated at len(initial_ys)
    #y_bounds: array-like of two floats or array-like of array-likes of two floats. If the former, is a uniform (upper bound, lower bound) for all y-values. If the latter, is an individualized upper/lower bound for each y.
    #x_bounds: array-like of two floats or array-like of array-likes of two floats. If the former, is a uniform (upper bound, lower bound) for all x-values. If the latter, is an individualized upper/lower bound for each x.
    #num_intervals: integer. If simType='fixedPoint', is the number of intervals. If simType='independent', is not used.
    #substrates: list of dictionaries. Each dictionary should have keys {'materialCall':callable taking sim_params dictionary as argument and returning material properties dictionary,'thickness':thickness of the substrate}. Substrates applied after rightmost layer but before
    #            transmission medium, and have no dependence on optimization parameters. They are applied in the order in which they appear in the list.
    #superstrates: list of dictionaries. Same format as substrates. Applied in the order in which they appear, e.g. the last element of superstrates is applied immediately before the first layer, the second-to-last element immediately before the last element, etc.
    #When doing a simulation rather than an optimization, make sure to provide the list of desired xs/ys 
    def set_structure(self,simType,num_layers,initial_layer_thickness,initial_ys=[],y_bounds=[0,1],x_bounds=[0,1],num_intervals=1,substrates=[],superstrates=[]):
        self.initial_layer_thickness,self.initial_ys,self.x_bounds,self.y_bounds,self.simType,self.rndx,self.rndy=[],[],[],[],simType,False,False
        if hasattr(initial_layer_thickness,'__iter__'):#If the user has supplied individual x values
            self.initial_layer_thickness=initial_layer_thickness#Use them
            if len(initial_ys)!=0:#If the user also provided individual y-values, we need to make sure the two lists are self consistent, with the y-values taking precidence
                initial_layer_thickness=initial_layer_thickness[:(len(initial_ys)+1+int(simType=='fixedPoint'))]#There should be no more x values than there are y-values, unless simType=='fixedPoint' in which case there should be exactly one more
                while len(initial_layer_thickness)!=(len(initial_ys)+int(simType=='fixedPoint')):#While we have fewer x-values than we should
                    initial_layer_thickness.append(initial_layer_thickness[-1])#Copy the last element of the x-list until we don't
            if simType=='fixedPoint':#If we're doing fixed-point optimization/simualtion, the number of intervals is the number of layer_thickness elements
                num_intervals=len(initial_layer_thickness)
            else:#If we're doing independent optimization/simulation, the number of layers is the number of layer_thickness elements
                num_layers=len(initial_layer_thickness)
        else:#If the user provided a single initial thickness
            if simType=='independent':#If the simulation is independent, make the initial x-list by randomizing a vector summing to the initial thickness
                for i in range(num_layers):
                    self.initial_layer_thickness.append(random())
                self.initial_layer_thickness=list(np.array(self.initial_layer_thickness)/sum(self.initial_layer_thickness)*initial_layer_thickness)
            if simType=='fixedPoint':#If the simulation is fixed-point, make the intial x-list by copying this number num_intervals times
                for i in range(num_intervals):
                    self.initial_layer_thickness.append(initial_layer_thickness)
        self.num_layers,self.num_intervals,self.substrates,self.superstrates=num_layers,num_intervals,substrates,superstrates
        if hasattr(x_bounds[0],"__iter__"):#If the user has supplied individual x-bounds
            x_bounds=x_bounds[:len(self.initial_layer_thickness)]#Make sure there aren't more x-bounds than there are x-values
            while len(x_bounds)!=len(self.initial_layer_thickness):#Make sure there aren't fewer x-bounds than there are x-values
                x_bounds.append(x_bounds[-1])
            self.x_bounds=x_bounds
        else:#If the user provided a single x-bound pair, copy it for all x-values
            for i in self.initial_layer_thickness:
                self.x_bounds.append(x_bounds)
        if hasattr(y_bounds[0],"__iter__"):#dido for the y-bounds
            y_bounds=y_bounds[:(len(self.initial_layer_thickness)-int(self.simType=='fixedPoint'))]
            while len(y_bounds)!=(len(self.initial_layer_thickness)-int(self.simType=='fixedPoint')):
                y_bounds.append(y_bounds[-1])
            self.y_bounds=y_bounds
        else:
            for i in range(len(self.initial_layer_thickness)-int(self.simType=='fixedPoint')):
                self.y_bounds.append(y_bounds)
        if len(initial_ys)==0:#If the user did not supply y-values, autogenerate them
            self.reroll_y()
            self.rndy=True
        else:
            self.initial_ys=initial_ys
        return
    
    def reroll_y(self):
        del self.initial_ys[:]
        for i in range(len(self.initial_layer_thickness)-int(self.simType=='fixedPoint')):#If we're doing fixed-point optimization, there is one more x-value than y-value since the last interval terminates in paramN
            self.initial_ys.append(uniform(self.y_bounds[i][0],self.y_bounds[i][1]))
        return
        

    #The partition of the space into layers can change in some circumstances, particularly between grand iterations. It doesn't take long to repartition the space, so I'm just doing it every iteration even though it is mildly inefficient. 
    #The number of iterations until completion rarely tops 300, so I don't think this is gonna be a bottleneck
    def _updateLayers(self,x):
        if self.simType=='independent':
            if self.var_x and self.var_y:
                self.x=x[:int(len(x)/2)]
                self.y=x[int(len(x)/2):]
            elif self.var_x:
                self.x=x
            elif self.var_y:
                self.y=x
            self.xLayer=self.x
            self.yLayer=self.y
        else:
            del self.xParamCoords[:]
            #For fixed-point optimization, x parameters = length of each interval. The last interval terminates in paramN, a fixed y-value, so there is one less y param than x param
            #xLayer=[l0,l1,...,l_{N-1}]
            #yLayer=[y0,y1,...,y_{N-1}]
            if self.var_x and self.var_y:
                self.x=x[:int((len(x)+1)/2)]
                self.y=x[int((len(x)+1)/2):]
            elif self.var_x:
                self.x=x
            elif self.var_y:
                self.y=x
            self.y_aux=[0]+list(self.y)+[0]
            self.xLayer,self.yLayer=[],[]
            for m in range(len(self.y_aux)-1):
                for j in range(self.num_layers):
                    self.xLayer.append(self.x[m]/self.num_layers)#length of sublists is constant within each interval, but depends upon the length of the interval
                    self.yLayer.append(self.y_aux[m]+(self.y_aux[m+1]-self.y_aux[m])/self.num_layers*(j+1))#y's of each interval stair-step from y of leftmost fixed point to y of rightmost. Last y in interval = rightmost fixed point's y
            return


    #By far the most convoluted part of this program. This is going to take some explaining.
    def _completeDerivative(self,dtmdxj,dtmdyj,dtmdyjp1,j,num_sup):
        Np=len(self.x)#Number of x-optParams
        if self.simType=='independent':#If each layer has a dedicated subset of optParams, it's easy
            self.all_dTdphis[-1][j][j+num_sup+1]+=dtmdxj#There is one more layer/transfer matrix than optParam. Layer j depends upon x_{j}, since both start at index 0; add dT(j+1)j/dx_{j}, 
            self.all_dTdphis[-1][j+Np][j+num_sup+1]+=dtmdyj#Do the same for dT(j+1)j/dy_{j}
            self.all_nonzero_dTdphi[-1][j].add(j+num_sup+1)#The derivative of the jth transfer matrix wrt the jth optParam is now nonzero. Make note of this.
            self.all_nonzero_dTdphi[-1][j+Np].add(j+num_sup+1)
            self.all_dTdphis[-1][j+1+Np][j+num_sup+1]+=dtmdyjp1#Do the same for dT(j+1)j/dy_{j+1}
            self.all_nonzero_dTdphi[-1][j+1+Np].add(j+num_sup+1)#The derivative of the jth transfer matrix wrt the j+1st optParam is now nonzero. Make note of this.
        else:#If each layer depends upon some subset of fixed points, it's hard. Let there be M x-optParams x_{m} and M-1 y-optParams y_{m}. let there be J sublayers in each of the N intervals
            k=j+1
            iv=int((j-j%self.num_layers)/self.num_layers)#This is the factor which determines on which optParams m x_{j},y_{j} depend
            ivp1=int((k-k%self.num_layers)/self.num_layers)#Dido for y_{j+1}
            dyjdyL=(self.num_layers-1-j%self.num_layers)/self.num_layers#This is a factor used in the derivative y_{j}/y_{m}
            dyjdyR=(1+j%self.num_layers)/self.num_layers#Dido for dy_{j}/dy_{m+1}
            dyjp1dyL=(self.num_layers-1-k%self.num_layers)/self.num_layers#Dido for dy_{j+1}/dy_{m}
            dyjp1dyR=(1+k%self.num_layers)/self.num_layers#Dido for dy_{j+1}/dy_{m+1}
            self.all_dTdphis[-1][iv][j+num_sup+1]+=dtmdxj/self.num_layers#xj will always depend on the x_{m} of the interval in which it lives. This interval is m=iv. 
            self.all_nonzero_dTdphi[-1][iv].add(j+num_sup+1)
            if iv>=1:#If this is the second or later interval
                self.all_dTdphis[-1][iv-1+Np][j+num_sup+1]+=dtmdyj*dyjdyL#Then yj will depend upon it's interval's leftmost y_{m}
                self.all_nonzero_dTdphi[-1][iv-1+Np].add(j+num_sup+1)
            if iv<len(self.x)-1:#If this is not the last interval
                self.all_dTdphis[-1][iv+Np][j+num_sup+1]+=dtmdyj*dyjdyR#Then yj will depend upon it's interval's rightmost y_{m}
                self.all_nonzero_dTdphi[-1][iv+Np].add(j+num_sup+1)
            if ivp1>=1:#If this is the second or later interval
                self.all_dTdphis[-1][ivp1-1+Np][j+num_sup+1]+=dtmdyjp1*dyjp1dyL#Then yjp1 will depend upon it's interval's leftmost y_{m}
                self.all_nonzero_dTdphi[-1][ivp1-1+Np].add(j+num_sup+1)
            if ivp1<len(self.x)-1:#If this is not the last interval
                self.all_dTdphis[-1][ivp1+Np][j+num_sup+1]+=dtmdyjp1*dyjp1dyR#Then yjp1 will depend upon it's interval's rightmost y_{m}
                self.all_nonzero_dTdphi[-1][ivp1+Np].add(j+num_sup+1)
        return

    def globalBoundaries(self,simDict):#Wrapper to give to agnostic_invDes. Computes the global boundary matrices. Also does builds incidental lists like customInput and the material parameters
        del self.all_mat1params[:]
        del self.all_mat2params[:]
        del self.all_param0s[:]
        del self.all_paramNs[:]
        del self.all_global_bcs[:]
        del self.all_custom_input[:]
        for tv in self.third_vars:
            for k in range(len(self.simPoints)):
                self.sim_params['third_vars'],self.sim_params['simPoint'],self.sim_params['callPoint']=tv,self.simPoints[k],self.callPoints[k]#Update sim_params dictionary
                mat1params,mat2params,param0,paramN=self.mat1Call(self.sim_params),self.mat2Call(self.sim_params),self.param0Call(self.sim_params),self.paramNCall(self.sim_params)#Get the material parameters so we only have to do it once
                self.all_mat1params.append(deepcopy(mat1params))#Add the material parameters to relevent lists for later use. List searching is faster than recomputing every set of params.
                self.all_mat2params.append(deepcopy(mat2params))
                self.all_param0s.append(deepcopy(param0))
                self.all_paramNs.append(deepcopy(paramN))
                self.all_custom_input.append(self.customInputCall(self.sim_params))
                self.all_global_bcs.append(self.pp.global_bcs(param0,paramN,self.sim_params,mat1params,mat2params))#Grab the global boundary matrices from the physics package
                
        return self.all_global_bcs

    def optFactors(self):#Wrapper to generate the cf factors/custom inputs. These are not needed for non-optimziation sims, so this is a seperate function
        del self.all_cf_factors[:]
        for tv in self.third_vars:
            for k in range(len(self.simPoints)):
                self.sim_params['third_vars'],self.sim_params['simPoint'],self.sim_params['callPoint']=tv,self.simPoints[k],self.callPoints[k]#Update sim_params dictionary
                self.all_cf_factors.append(self.cfFactorCall(self.sim_params))
                
        return
    
    def paramPruner(self,params,it):
        self.cur_x,self.cur_y=self._isolate_xy(params)
        self.cur_x,self.cur_y,self.x_bounds,self.y_bounds=self.paramPrunerCall(self.cur_x,self.cur_y,self.x_bounds,self.y_bounds,it)
        return self._merge_xy(self.cur_x,self.cur_y),list(self.x_bounds)*int(self.var_x)+list(self.y_bounds)*int(self.var_y)
            
    #This gets a tad complicated because the nature of the interface matrix between semiinfinite media and finite layers is physics-dependent.
    def transferMatrices(self,simDict):#Wrapper to give to agnostic_invDes. Computes the transfer matrices
        del self.all_tms[:]
        del self.all_tracked_tm_info[:]
        self._updateLayers(simDict['parameters'])#Re-partition the layers. Doesn't take too long and it's nice to not have to keep track of when the structure has changed
        i=0
        for tv in self.third_vars:#Run through third variables
            for k in range(len(self.simPoints)):#Run through simulation points
                self.sim_params['third_vars'],self.sim_params['simPoint'],self.sim_params['callPoint']=tv,self.simPoints[k],self.callPoints[k]#Update sim_params dictionary
                mat1params,mat2params,param0,paramN=self.all_mat1params[i],self.all_mat2params[i],self.all_param0s[i],self.all_paramNs[i]#Grab material parameter dictionaries
                self.all_tms.append([])#All_tms will have one list per third_var-simPoint pair
                self.all_tracked_tm_info.append([])
                tracked_info={}#Initialize tracked_info as an empty dictionary
                if len(self.superstrates)>0:#If we have superstrates
                    tm,tracked_info=self.pp.left_tm(1,self.sim_params,param0,self.superstrates[0]['material_call'](self.sim_params),self.superstrates[0]['material_call'](self.sim_params))#Do the incidence medium interface with the first superstrate
                    self.all_tracked_tm_info[-1].append(tracked_info)
                    self.all_tms[-1].append(tm)
                    for m in range(len(self.superstrates)):#Run through all the superstrates
                        ss=self.superstrates[m]
                        if m<len(self.superstrates)-1:#If there is another superstrate after this one, use it for the interface matrix
                            tm,tracked_info=self.pp.interface_tm(ss['thickness'],1,1,self.sim_params,tracked_info,ss['material_call'](self.sim_params),ss['material_call'](self.sim_params),self.superstrates[m+1]['material_call'](self.sim_params),self.superstrates[m+1]['material_call'](self.sim_params))
                        else:#If there is not another superstrate after this one, use the first optimization layer for the interface matrix
                            tm,tracked_info=self.pp.interface_tm(ss['thickness'],1,self.yLayer[0],self.sim_params,tracked_info,ss['material_call'](self.sim_params),ss['material_call'](self.sim_params),mat1params,mat2params)
                        self.all_tms[-1].append(tm)
                        self.all_tracked_tm_info[-1].append(tracked_info)
                else:#If there are no superstrates, do the incidence medium interface with the first simulation layer
                    tm,tracked_info=self.pp.left_tm(self.yLayer[0],self.sim_params,param0,mat1params,mat2params)
                    self.all_tracked_tm_info[-1].append(tracked_info)
                    self.all_tms[-1].append(tm)
                x,y=self.xLayer[0],self.yLayer[0]#At this point, we are at the first field point inside the optimization region
                for j in range(len(self.xLayer)-1):#Run through each optimziaiton layer, minus the last one
                    xp1,yp1=self.xLayer[j+1],self.yLayer[j+1]#Get the x/y of the next optimzation layer
                    tm,tracked_info=self.pp.tm(x,y,yp1,self.sim_params,tracked_info,mat1params,mat2params)#Build the transfer matrix
                    self.all_tms[-1].append(tm)
                    self.all_tracked_tm_info[-1].append(tracked_info)
                    x,y=xp1,yp1#Update x/y for the next iteration
                #At this point, x,y are the x/y of the last optimization layer, the transfer matrix of which has not yet been computed
                if len(self.substrates)>0:#If we have substrates
                    tm,tracked_info=self.pp.interface_tm(x,y,1,self.sim_params,tracked_info,mat1params,mat2params,self.substrates[0]['material_call'](self.sim_params),self.substrates[0]['material_call'](self.sim_params))#Build the last-optimization-layer-2-first-substrate matrix
                    self.all_tracked_tm_info[-1].append(tracked_info)
                    self.all_tms[-1].append(tm)
                    for m in range(len(self.substrates)):#Run through all the substrates
                        ss=self.substrates[m]
                        if m<len(self.substrates)-1:#If this is not the last substrate
                            tm,tracked_info=self.pp.interface_tm(ss['thickness'],1,1,self.sim_params,tracked_info,ss['material_call'](self.sim_params),ss['material_call'](self.sim_params),self.substrates[m+1]['material_call'](self.sim_params),self.substrates[m+1]['material_call'](self.sim_params))#Connect this and the next substrate
                            self.all_tracked_tm_info[-1].append(tracked_info)
                        else:#If it is the last substrate
                            tm=self.pp.right_tm(ss['thickness'],1,self.sim_params,tracked_info,ss['material_call'](self.sim_params),ss['material_call'](self.sim_params),paramN)#Connect it and the transmission medium
                        self.all_tms[-1].append(tm)
                else:#If there are no substrates, connect the last optimizaiton layer to the transmission medium
                    tm=self.pp.right_tm(x,y,self.sim_params,tracked_info,mat1params,mat2params,paramN)
                    self.all_tms[-1].append(tm)
                i+=1
        return self.all_tms

    #If you thought transferMatrices() was bad, wait till you see this one. This is by far the most complicated part of the entire agnostic inverse design framework
    def transferMatrices_gradPhi(self,simDict):#Wrapper to give to agnostic_invDes. Computes the derivatives of the transfer matrices
        del self.numpy_zv[:]
        del self.all_dTdphis[:]
        del self.all_nonzero_dTdphi[:]
        for tm in self.all_tms[0]:#By default, the derivative of every transfer matrix will be zero unless determined otherwise. Build this vector of zero matrices
            self.numpy_zv.append(np.zeros((len(tm),len(tm)),dtype=np.complex64))
        i=0
        self.chosen_simPoints=set()#For stochasitc gradient descent, we don't want to use every simPoint for gradient computation. Randomly select those that will be used.
        simPointInds=range(len(self.simPoints))
        if self.stochastic_generation==0:
            self.chosen_simPoints=set(simPointInds)
        else:
            while len(self.chosen_simPoints)!=self.stochastic_generation:
                self.chosen_simPoints.add(choice(simPointInds))
        for tv in self.third_vars:
            for k in range(len(self.simPoints)):
                self.sim_params['third_vars'],self.sim_params['simPoint'],self.sim_params['callPoint']=tv,self.simPoints[k],self.callPoints[k]#Update sim_params dictionary
                mat1params,mat2params,param0,paramN,tracked_info,tms=self.all_mat1params[i],self.all_mat2params[i],self.all_param0s[i],self.all_paramNs[i],self.all_tracked_tm_info[i],self.all_tms[i]
                self.all_dTdphis.append([])
                self.all_nonzero_dTdphi.append([])
                for p in range(len(self.x)+len(self.y)):#First, fill out dTdphi with all zeros, one vector of zero matrices per optimizaiton parameter
                    self.all_dTdphis[-1].append(deepcopy(self.numpy_zv))
                    self.all_nonzero_dTdphi[-1].append(set())#Fill out list of nonzero dTdphis with empty sets
                if k in self.chosen_simPoints:#If this is a chosen simPoint
                    num_sup,num_sub,num_x=len(self.superstrates),len(self.substrates),len(self.x)
                    #agnostic_adjoint will be expecting one dT/dphi per T, even for the super/substrate transfer matrices. These of course do not depend upon optimization parameters, except for maybe the last superstrate which could depend on the first x/y via its interface
                    if num_sup>0:#If we have a superstrate, then the incidence matrix has no optParam dependence, but the last superstrate matrix might
                        ss=self.superstrates[-1]
                        dtmdxj,dtmdyj,dtmdyjp1=self.pp.d_interface_tm(ss['thickness'],1,self.yLayer[0],self.sim_params,tracked_info[num_sup],tms[num_sup],ss['material_call'](self.sim_params),ss['material_call'](self.sim_params),mat1params,mat2params)#Get the partials wrt all layer parameters
                    else:#If we don't have a superstrate, then the incidence matrix might have optParam dependence.
                        dtmdyjp1=self.pp.d_left_tm(self.yLayer[0],self.sim_params,tracked_info[0],tms[0],param0,mat1params,mat2params)
                    dyjp1dy0=1*int(self.simType=='independent')+1/self.num_layers*int(self.simType=='fixedPoint')#Only yjp1 could have optParam dependence (on the first y optParam), since xj,yj are fixed superstrate parameters. The exact dependence depends on simType  
                    self.all_dTdphis[-1][num_x][num_sup]+=dtmdyjp1*dyjp1dy0#Add the possibly nonzero derivative
                    self.all_nonzero_dTdphi[-1][num_x].add(num_sup)#Make a note that this one is nonzero
                    x,y=self.xLayer[0],self.yLayer[0]#We now go through the optimization layers
                    for j in range(len(self.xLayer)-1):
                        xp1,yp1=self.xLayer[j+1],self.yLayer[j+1]
                        dtmdxj,dtmdyj,dtmdyjp1=self.pp.dtm(x,y,yp1,self.sim_params,tracked_info[j+num_sup+1],tms[j+num_sup+1],mat1params,mat2params)#Find the partials wrt layer parameters. The j+num_sup+1 is because j=0 is the first optLayer, but there are superstrates and the incidence matrix before that
                        self._completeDerivative(dtmdxj,dtmdyj,dtmdyjp1,j,num_sup)#Complete the derivatives by chaining the derivative of the layer parameters wrt the optimizaiton parameters. This is where a lot of the complexity is hidden
                        x,y=xp1,yp1
                    #At this point x,y are the x/y of the last optimization layer. j is the index of the second-to-last optimization layer in yLayers/xLayers
                    j+=1
                    if num_sub>0:#If we have a substrate, then the transmission matrix has no optParam dependence, but the last optimization layer matrix is interfaced with the first substrate
                        ss=self.substrates[0]
                        dtmdxj,dtmdyj,dtmdyjp1=self.pp.d_interface_tm(x,y,1,self.sim_params,tracked_info[j+num_sup+1],tms[j+num_sup+1],mat1params,mat2params,ss['material_call'](self.sim_params),ss['material_call'](self.sim_params))#Get the partials wrt all layer parameters
                    else:#If we don't have a superstrate, then the transmission matrix might have optParam dependence.
                        dtmdyjp1=self.pp.d_right_tm(x,y,self.sim_params,tracked_info[-1],tms[-1],mat1params,mat2params,param0)
                    dyjdyN=1*int(self.simType=='independent')+0*int(self.simType=='fixedPoint')#yj could have optParam dependence (on the last y optParam). The exact dependence depends on simType
                    dxjdxN=1*int(self.simType=='independent')+(1/self.num_layers)*int(self.simType=='fixedPoint')#xj could have optParam dependence (on the last x optParam). The exact dependence depends on simType
                    self.all_dTdphis[-1][num_x-1][num_sup+j+1]+=dtmdxj*dxjdxN#Add the possibly nonzero derivatives
                    self.all_nonzero_dTdphi[-1][num_x-1].add(num_sup+j+1)#Make a note that this one is nonzero
                    self.all_dTdphis[-1][-1][num_sup+j+1]+=dtmdyj*dyjdyN#Add the possibly nonzero derivatives
                    self.all_nonzero_dTdphi[-1][-1].add(num_sup+j+1)#Make a note that this one is nonzero
                else:
                    for k in range(len(self.x)+len(self.y)):#Bit of an oversight on my part, if nonsero_dTdphi is empty, it is assumed none are nonzero. All are zero for non-chosen simPoints, so add one dummy index for these. This means the computer will do
                                                            #one more matrix multiplication than it needs to per non-chosen simPoint, which is not too bad.
                        self.all_nonzero_dTdphi[-1][k].add(0)
                if not self.var_x:#Prune the derivatives if one of x or y is not an optParam
                    del self.all_dTdphis[-1][:len(self.x)]
                    del self.all_nonzero_dTdphi[-1][:len(self.x)]
                if not self.var_y:
                    del self.all_dTdphis[-1][len(self.x):]
                    del self.all_nonzero_dTdphi[-1][len(self.x):]
                i+=1
        return self.all_dTdphis,self.all_nonzero_dTdphi

    def _isolate_xy(self,parameters):
        if self.var_x and self.var_y:
            xdiv=int((len(parameters)+int(self.simType=='fixedPoint'))/2)
            x,y=parameters[:xdiv],parameters[xdiv:]
        elif self.var_x:
            x,y=parameters,self.cur_y
        elif self.var_y:
            x,y=self.cur_x,parameters
        return x,y
        
    def _merge_xy(self,x,y):
        return list(x)*int(self.var_x)+list(y)*int(self.var_y)

    def costFunction(self,simDict):#Wrapper to give to agnostic_invDes. Computes the cost function
        params=simDict['parameters']
        x,y=self._isolate_xy(params)
        self.optFactors()#Get the cf factors and custom inputs
        rvs=self.pp.costFunc(self.simPoints,self.callPoints,self.third_vars,x,y,self.all_mat1params,self.all_mat2params,self.all_param0s,self.all_paramNs,simDict['fields'],simDict['transferMatrices'],self.all_global_bcs,self.all_cf_factors,self.all_scheduler_factors,self.all_custom_input)
        if not hasattr(rvs,'__iter__'):
            cfVal=rvs
            self.cfoutputs,Lphys,Lreg={},0,0
        elif len(rvs)==2:
            cfVal,self.cfoutputs=rvs
            Lphys,Lreg=0,0
        else:
            cfVal,self.cfoutputs,Lphys,Lreg=rvs
        return cfVal,Lphys,Lreg
    
    def costFunction_gradPhi(self,simDict):#Wrapper to give to agnostic_invDes. Computes the partial d[cost function]/d[optParams]
        params=simDict['parameters']
        x,y=self._isolate_xy(params)
        dl=self.pp.d_costFunc_phi(self.simPoints,self.callPoints,self.third_vars,x,y,self.all_mat1params,self.all_mat2params,self.all_param0s,self.all_paramNs,simDict['fields'],simDict['transferMatrices'],self.all_global_bcs,self.all_cf_factors,self.all_scheduler_factors,self.all_custom_input,self.cfoutputs,self.chosen_simPoints)
        if not self.var_x:
            dl=dl[len(self.cur_x):]
        if not self.var_y:
            dl=dl[:len(self.cur_x)]
        return dl

    def costFunction_gradE(self,simDict):#Wrapper to give to agnostic_invDes. Computes the partial d[cost function]/d[fields]
        params=simDict['parameters']
        x,y=self._isolate_xy(params)
        '''
        print('sim_params: '+str(self.sim_params))
        print('all_cf_factors: '+str(self.all_cf_factors))
        print('all_scheduler_factors: '+str(self.all_scheduler_factors))
        print('all_custom_input: '+str(self.all_custom_input))
        print('x: '+str(x))
        print('y: '+str(y))
        '''
        return self.pp.d_costFunc_fields(self.simPoints,self.callPoints,self.third_vars,x,y,self.all_mat1params,self.all_mat2params,self.all_param0s,self.all_paramNs,simDict['fields'],simDict['transferMatrices'],self.all_global_bcs,self.all_cf_factors,self.all_scheduler_factors,self.all_custom_input,self.cfoutputs,self.chosen_simPoints)

    def evo_preamble(self):#Preamble to the evo dictionary, which contains all information on the options chosen by the user for this optimization so it can be reconstructed later if needed
        func_names=[]
        for func in [self.mat1Call,self.mat2Call,self.param0Call,self.paramNCall,self.cfFactorCall,self.schedulerCall,self.customInputCall]:
            name=str(func)
            name_lis=name.split(' ')
            func_names.append(name_lis[1])
        self.evo[-1]={'physics':self.physics,'third_vars':self.third_vars,'simPoints':self.simPoints,'simScale':self.simScale,'simType':self.simType,'num_intervals':self.num_intervals,
                      'discretization_map':self.discretization_map,'x_bounds':self.x_bounds,'y_bounds':self.y_bounds,'num_layers':self.num_layers,'substrates':self.substrates,
                      'superstrates':self.superstrates,'mat1_name':func_names[0],'mat2_name':func_names[1],'param0_name':func_names[2],'paramN_name':func_names[3],'cf_function':func_names[4],
                      'scheduler':func_names[5],'customInput':func_names[6],'evo_preamble_tag':self.evo_preamble_tag,'ftol':self.ftol,'gtol':self.gtol,'name':self.save_name}
        return
    
    #Prints the relevent bits of the evo preamble, so the user knows what physics functions to use when doing analysis.
    def print_evo_info(self,sparse=False):
        print('Physics package: '+self.evo[-1]['physics'])
        print('Preamble tag: '+str(self.evo[-1]['evo_preamble_tag']))
        print('Simulation type: '+self.evo[-1]['simType'])
        print('Material 1 function name: '+self.evo[-1]['mat1_name'])
        print('Material 2 function name: '+self.evo[-1]['mat2_name'])
        print('param0 function name: '+self.evo[-1]['param0_name'])
        print('paramN function name: '+self.evo[-1]['paramN_name'])
        print('customInput function name: '+self.evo[-1]['customInput'])
        print('Third Variables: '+str(self.evo[-1]['third_vars']))
        print('Superstrates: '+str(self.evo[-1]['superstrates']))
        print('Substrates: '+str(self.evo[-1]['substrates']))
        print('Number of grand iterations: '+str(len(self.grand_iters)-1))
        iter_lis=[]
        for i in self.grand_iters:
            if i>=0:
                iter_lis.append(len(self.evo[i]))
        print('Number of iterations in each grand iteration: '+str(iter_lis))
        if not sparse:
            print('Optimization time: '+str(self.evo[-1]['optimization_time']))
            print('cf function name: '+self.evo[-1]['cf_function'])
            print('Scheduler name: '+self.evo[-1]['scheduler'])
            print('x_bounds[0]: '+str(self.evo[-1]['x_bounds'][0]))
            print('y_bounds[0]: '+str(self.evo[-1]['y_bounds'][0]))
            print('Simulation range, resolution: '+str([self.evo[-1]['simPoints'][0],self.evo[-1]['simPoints'][-1]])+', '+str(len(self.evo[-1]['simPoints'])))
            print('simScale, conversion from simulation unit to SI units: '+str(self.evo[-1]['simScale']))
            print('ftol: '+str(self.evo[-1]['ftol']))
            print('gtol: '+str(self.evo[-1]['gtol']))
            print('Number of layers: '+str(self.evo[-1]['num_layers']))
            print('Number of intervals (fixedPoint only): '+str(self.evo[-1]['num_intervals']))
            print('Discretization map: '+str(self.evo[-1]['discretization_map']))
            
        return

    #simType: string, either "independent", in which the possible optimization parameters are the thickness x and some other physical parameter y (reractive index, ratio, etc.) of each layer; or "fixedPoint", in which the optParams are the y-values of fixed points and 
    #         the x values are lengths of intervals between each fixed point. The points are linearly interpolated and then broken into num_layers steps within each interval for simulation.
    #simRange: tuple, the minimum/maximum of the simulation range
    #simResolution: integer, the number of simulation points
    #third_variables: list of array-likes or dictionaries, extra sweep parameters. These could, for example, be a set of temperature-pressure coordinates, where you want to sweep frequency at each temperature/pressure. Simulations will be carried out at each 
    #                 element of third_vars in sequence,and the current third_vars of a particular iteration will be sent to material functions, the cost function, and matrix builders in the sim_params dictionary. Must have at least one element
    #discretization_map: list of dictionaries. Dictionaries should have keys {'var_x':bool, allow x to be an optParam; 'var_y':bool, allow y to be an optParam; 'scramble_x':[bool, whether to scramble; float, the scramble factor],
    #                                                                         'scramble_y':[bool, whether to scramble; float, the scramble factor],'merge_neighbors':[bool,tolerance],'fully_discretize':bool}
    #                    Each dictionary describes a single "grand_iteration"; the scheduler is also called and weights on regularization terms in the cost function are also updated each grand_iteration. The order of operations in a single grand iteration is as follows:
    #                       1.Scheduler called, regularization weights updated
    #                       2. Optimization performed, with x, y, or both used as optParams according to the current discretization dictionary
    #                       3. x/y values scrambled if the option was selected. Each x/y is randomly increased or decreased by between 0% and 100*scramble_factor%, subject to x_bounds/y_bounds.
    #                       4. The structure is forcibly discretized if the option was selected, each y forced to either its lower bound or upper bound, whichever is closer to the current value.
    #                       5. Neighbors merged if the option was selected. Neighboring layers/fixedPoints whose y-values are withing tolerance of eachother are merged into a single layer/fixedPoint
    #                       6. The resultant structure is passed to the next grand_iteration
    #simScale: float, the factor by which the given simRange numbers should be multiplied to convert them to SI units. For example, if the simulation is from 1um to 5um wavelength, simRange=[1,5] and simScale=1E-6, if you wish to do the simulating in um.
    #ftol, gtol: floats, the function change tolerance and gradient tolerance of the optimizer. See the scipy.minimize documentation
    #save_name: string, the name of the output pickle file containing the evolution dictionary. Do not include file extension
    #save_directory: string, the directory to which the pickle file should be saved. Include final seperation character.
    #stochastic generation: integer, the number of simPoints to use for gradient calculation. Points will be chosen at random each iteration. Set equal to zero to use all points for the gradient.
    #logSim: bool, whether to distribute simPoints logarithmically within range
    #verbose: integer, how often to display updates on convergence. Set equal to 0 for silent mode.
    #evo_preamble_tag: anything, will be added to the evolution dictionary's preamble under key 'evo_preamble_tag'
    def run_optimization(self,simRange,simResolution,third_variables=['n/a',],discretization_map=[],simScale=1,ftol=1E-9,gtol=1E-5,save_name='unified_test',save_directory='',save=True,
                         logSim=False,verbose=1,evo_preamble_tag='',presearch_threshold=np.inf):
        stochastic_generation=0#I kept forgetting to disable stochastic descent and my off-brand version of it never really worked anyway, so the SGD option has lost its existance privileges.
        self.simRange,self.evo_preamble_tag,self.third_vars,self.simScale,self.ftol,self.gtol,self.save_name,self.save_directory,self.stochastic_generation,self.verbose,self.logSim=simRange,evo_preamble_tag,third_variables,simScale,ftol,gtol,save_name,save_directory,stochastic_generation,verbose,logSim
        if len(simRange)>2:
            self.simPoints=np.array(simRange)
            self.simResoluion=len(simRange)
        elif logSim:#Generate the simulation points
            self.simPoints=np.logspace(np.log10(simRange[0]),np.log10(simRange[1]),simResolution)
        else:
            self.simPoints=np.linspace(simRange[0],simRange[1],simResolution)
        self.callPoints=self.simPoints*simScale#Generate the call points, in SI units, for use by material functions
        if len(discretization_map)==0:#Set the default grand_iteration parameters if none were provided
            discretization_map.append({'var_x':False,'var_y':True,'scramble_x':[False,0.01],'scramble_y':[False,0.01],'merge_neighbors':[False,0.05], 'fully_discretize':False,'purge_ablated':[False,0.005]})
        self.discretization_map=discretization_map
        self.it_params={'grand_iteration':0,'previous_costFunction':'N/A'}#Initialize the iteration parameters dictionary
        if not self.dynamic_scheduling:
            self.all_scheduler_factors=self.schedulerCall(self.it_params)#Get the intial scheduler factors
            sc=''
        else:
            self.all_scheduler_factors=self.schedulerCall(np.inf,np.inf,-1,self.initial_layer_thickness,self.initial_ys,[])
            sc=self.scheduler
        if callable(self.paramPrunerCall):
            ppc=self.paramPruner
        else:
            ppc=''
        self.evo_preamble()#Generate the evo dicitonary preamble
        self.grand_iter=0#Grand_iteration 0 is the initial values and the initial cost function. Because minimize only calls callback after the first optimziation step, this must be done seperately
        self.aid=agnostic_invDes(self.costFunction,self.costFunction_gradPhi,self.costFunction_gradE,self.globalBoundaries,self.transferMatrices,self.transferMatrices_gradPhi,sc,ppc)#Instantiate the agnostic inverse designer
        self.aid.debug_verbosity=self.debug_verbosity
        self.var_x,self.var_y=True,True#For grand_iteration 0, set both x/y as variable
        if verbose>0:
            print('Beginning presearch')
        l,dl=self.aid._costFunc_wrapper(list(self.initial_layer_thickness)+list(self.initial_ys))#Grab the cost function of the initial paramters
        while l>=presearch_threshold and self.rndy:
            self.reroll_y()
            l,dl=self.aid._costFunc_wrapper(list(self.initial_layer_thickness)+list(self.initial_ys))#Grab the cost function of the initial paramters
        if verbose>0:
            print('Finished presearch')
        self.evo[0]=[{'costFunction':l,'costFunction-Physical': self.aid.Lphys,'costFunction-Regularization': self.aid.Lreg,'parameters':list(self.initial_layer_thickness)+list(self.initial_ys)}]#Add the evo entry
        self.grand_iter=1#grand_iteration 1 is the first optimization grand iteration
        self.it_params={'grand_iteration':1,'previous_costFunction':self.evo[0][0]['costFunction']}
        self.cur_x,self.cur_y=list(self.initial_layer_thickness),list(self.initial_ys)#Set the current x/y parameter values
        if not self.dynamic_scheduling:
            self.all_scheduler_factors=self.schedulerCall(self.it_params)#Get the first optimization iteration scheduler factors
        else:
            self.all_scheduler_factors=self.schedulerCall(np.inf,l,0,self.initial_layer_thickness,self.initial_ys,self.all_scheduler_factors)
        optimization_start=datetime.now()#The actual optimization starts now
        for disc_dict in self.discretization_map:#Run through the discretization steps
            cur_params,cur_bounds,self.var_x,self.var_y=list(self.cur_x)*int(disc_dict['var_x'])+list(self.cur_y)*int(disc_dict['var_y']),self.x_bounds*int(disc_dict['var_x'])+self.y_bounds*int(disc_dict['var_y']),disc_dict['var_x'],disc_dict['var_y']#Select the appropriate varaibles as optParams this step
            self.res=self.aid.optimize(cur_params,verbose=verbose,ftol=ftol,gtol=gtol,bounds=cur_bounds)#Run the optimization
            self.evo[self.grand_iter]=[]
            for iter in range(len(self.res.evo)):
                ce=self.res.evo[iter]
                self.evo[self.grand_iter].append({'costFunction': ce['costFunction'],'costFunction-Physical': ce['costFunction-Physical'],'costFunction-Regularization': ce['costFunction-Regularization'],
                                                  'parameters':list(deepcopy(self.cur_x))*int(not self.var_x)+list(ce['parameters'])+list(deepcopy(self.cur_y))*int(not self.var_y)})#In our evo_dictionary, we want both x and y. So, add x/y if one of them was not an optParam this iteration
            self.cur_x,self.cur_y=self._isolate_xy(self.res.x)#Update cur_x and cur_y
            if disc_dict['scramble_x'][0]:#If we're scrambling x, do that
                self.scramble('x',disc_dict['scramble_x'][1])
            if disc_dict['scramble_y'][0]:#If we're scrambling y, do that
                self.scramble('y',disc_dict['scramble_y'][1])
            if disc_dict['fully_discretize']:#If we're forcibly discretizing, do that
                self.fully_discretize()
            if disc_dict['merge_neighbors'][0]:#If we're merging neighbors, do that
                self.merge_neighbors(disc_dict['merge_neighbors'][1])
            if disc_dict['purge_ablated'][0]:
                self.purge_ablated(disc_dict['purge_ablated'][1])
            self.grand_iter+=1
            self.it_params={'grand_iteration':self.grand_iter,'previous_costFunction':self.evo[self.grand_iter-1][-1]['costFunction']}#Update the scheduler factors for the next iteration
            if not self.dynamic_scheduling:#Only update scheduler weights outside of minimize if not doing dynamic scheduling
                self.all_scheduler_factors=self.schedulerCall(self.it_params)
        optimization_end=datetime.now()#The actual optimzation is now done. The rest is just clean-up
        self.evo[-1]['optimization_time']=optimization_end-optimization_start
        self.grand_iters=list(range(0,self.grand_iter+1))
        if verbose>0:
            print('Optimization complete. Discounting overhead, time elapsed was '+str(optimization_end-optimization_start))
        self.var_x,self.var_y=True,True#The final grand_iteration is the final structure, for easy access later
        l,dl=self.aid._costFunc_wrapper(list(self.cur_x)+list(self.cur_y))
        self.evo[self.grand_iter]=[{'costFunction':l,'costFunction-Physical': self.aid.Lphys,'costFunction-Regularization': self.aid.Lreg,'parameters':list(self.cur_x)+list(self.cur_y)}]
        if save:
            self.pu_pickle_evo(save_name,self.evo,dir=save_directory)#Save the evo dictionary
        return

    def merge_neighbors(self,tol):#Merge neighboring layers
        r=0
        while r<len(self.cur_y)-1:#Run through all y's
            if abs(self.cur_y[r]-self.cur_y[r+1])<=tol:#If this y and the next y are within tolerance of eachother
                self.cur_y[r]=(self.cur_y[r]+self.cur_y[r+1])/2#The new y-value is their average
                self.cur_x[r]+=self.cur_x[r+1]#The new x-value is the sum of this and the next x-value
                self.x_bounds[r]=[self.x_bounds[r][0],self.x_bounds[r][1]+self.x_bounds[r+1][1]]#Expand x_bounds to accomodate the new, enlarged x
                del self.cur_y[r+1]#Remove the now-merged next layer
                del self.cur_x[r+1]
                del self.x_bounds[r+1]
                del self.y_bounds[r+1]
                if self.simType=='independent':#Make note of the fact that we now have one less layer or interval
                    self.num_layers-=1
                else:
                    self.num_intervals-=1
            else:
                r+=1
        return

    def fully_discretize(self):#Forcibly discretize the structure. Intended to round off small non-discretness due to e.g. machine error. Discretization should be primarily accomplished through cost function regularization terms
        for r in range(len(self.cur_y)):#Run through each y
            mp=(self.y_bounds[r][0]+self.y_bounds[r][1])/2#Find the midpoint of that y's bounds
            if self.cur_y[r]>=mp:#If the y is greater than the midpoint, set it at the upper bound
                self.cur_y[r]=self.y_bounds[r][1]
            else:#Otherwise, set it at the lower bound
                self.cur_y[r]=self.y_bounds[r][0]
        return

    def scramble(self,which,factor):#Scramble x/y. Makes for a more stochastic dscent, with random kicks to the structure to (hopefully) knock it out of shallow minima. Can also knock it out of deep minima if you're not careful, so use sparingly
        if which=='x':#Can scramble either x or y
            for i in range(len(self.cur_x)):#Run through each optParam
                x=self.cur_x[i]
                cur_bnd=self.x_bounds[i]#Grab that optParam's bounds
                self.cur_x[i]=uniform(max(cur_bnd[0],x*(1-factor)),min(cur_bnd[1],x*(1+factor)))#Change optParam to some value between [current value]*(1-factor) and [current value]*(1+factor), subject to boundary constraints
        else:
            for i in range(len(self.cur_y)):
                y=self.cur_y[i]
                cur_bnd=self.y_bounds[i]
                self.cur_y[i]=uniform(max(cur_bnd[0],y*(1-factor)),min(cur_bnd[1],y*(1+factor)))
        return
    
    def purge_ablated(self,thresh):
        i=0
        while i<len(self.cur_x):
            if self.cur_x[i]<=thresh:
                self.cur_x=self.arraydel(self.cur_x,i)
                self.cur_y=self.arraydel(self.cur_y,i)
                self.x_bounds=self.arraydel(self.x_bounds,i)
                self.y_bounds=self.arraydel(self.y_bounds,i)
            else:
                i+=1
        return
                
            