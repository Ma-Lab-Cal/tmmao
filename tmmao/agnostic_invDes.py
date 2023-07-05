import numpy as np
from copy import deepcopy
from random import random, uniform, randint, choice
from scipy.optimize import minimize
from agnostic_linear_adjoint import agnostic_linear_adjoint
from datetime import datetime
import warnings

class termOpt(Exception):
    """Nothing to see here. Move along.
    
        Since I have very little control of scipy.minimize via the api, I had to get creative to allow for dynamic scheduling/parameter mutation during optimization. This is a special
        exception defined for that purpose. You as a user should never see it raised.
    """
    pass

class agnostic_invDes(agnostic_linear_adjoint):
    """Runs an optimization
    
    Subclasses:
        agnostic_linear_adjoint: Computes the adjoint fields and gradients.
    
    Attributes:
        simDict: Dictionary. Contains data used in current optimization iteration. Keys/entries are:
        
            {'fields':Array-like of array-likes of numpy arrays, such that simDict['fields'][n][j]=e_j for the nth simPoint,
             'transferMatrices': Array-like of array-likes of numpy arrays, such that simDict['transferMatrices'][n][j]=T^{j+1,j} for the nth simPoint
             'parameters': Array-like of floats. The current value of the optimization parameters such that simDict['parameters'][m]=phi_m. Note that during linesearch, this set of phi_m may differ from the actual current structure of the device; see minimize's L-BFGS-B documentation, such as it is,
             'previousCostFunction': Float. The previous iteration's cost function
             'iteration': Int. The current iteration.}
   
        res: optimizationResults instance. Houses the results of the optimization; these are updated dynamically during optimization as far as is possible. See optimizationResults for attributes.
        L: FLoat. The current cost function.
        Lphys: Float. The current value of the physics terms in the cost function, if provided by the user's cost function.
        Lreg: Float. The current value of the regularization terms in the cost function, if provided by the user's cost function.
        debug_verbosity: Bool. If True, provides increased verbosity for debugging purposes.
        bounds: Array-like of two-element array-likes. The current bounds of each optimization parameter, if doing bounded optimization.
        
    Methods:
        updatePhysicsFunctions: Updates some or all of the physics functions given to the class constructor at instantiation.
        callback: Callback function handed to minimize, called after every iteration. Updates res attributes, and interfaces with the scheduler and any parameter mutators. It can be conveient to call callback() manually to add a set of paramters to the
            evo dictionary outside of an optimization iteration, a strategy used in agnostic_director.
        optimize: Primary API for inverse design. Runs the optimization given initial parameters, and returns res.
        simulate: Primary API for simulation. Runs a simulation given a set of optimization parameters.
        
    Raises:
        RuntimeWarning: Warning raised by scipy.minimize if it judges convergence to be too slow. It thinks more or less everything is too slow, so ignore it unless you're actually experiencing convergence issues.
    """
    def __init__(self,costFunction,costFunction_gradPhi,costFunction_gradE,globalBoundaries,transferMatrices,transferMatrices_gradPhi,scheduler='',paramPruner=''):
        """Initializes instance of agnostic_invDes
        
        Args:
            costFunction: callable. Accepts simDict as argument, returns 
                L: Scalar. The cost function.
            costFunction_gradPhi: callable. Accepts simDict as argument, returns 
                dLdphi: Array-like of scalars. The partial derivatives of the cost function wrt the optimization parameters, such that dLdphi[m]=dL/dphi_m
            costFunction_gradE: callable. Accepts simDict as argument, returns
                dLde: Array-like of array-likes of numpy arrays or equivalent. The partial derivatives of the cost function wrt the fields, such that dLde[n][j]=dL/de_j for simPoint n.
            globalBoundaries: callable. Accepts simDict as argument, returns
                gbcs: Array-like of three-element array likes. The global boundaries, such that gbcs[n]=[A,B,c] for simPoint n, where A.e_0+B.e_N=c
            transferMatrices: callable. Accepts simDict as argument, returns
                all_tms: Array-like of array-likes of numpy arrays or equivalent. The transfer matrices, such that all_tms[n][j]=T^{j+1,j} for simPoint n.
            transferMatrices_gradPhi: callable. Accepts simDict as argument, returns
                dTdphi: Array-like of array-likes of array-likes of numpy arrays or equivalent. The partial derivatives of the transfer matrices wrt the optimization parameters, such that dTdphi[n][m][j]=dT^{j+1,j}/dphi_m for simPoint n.
            scheduler: callable or non-callable; optional. If callable, must accept scheduler(L_prev->float,L_cur->float, its->int) where L_prev is the previous cost function, L_cur the current cost function, and its the current iteration. 
                It should return nothing. The scheduler can dynamically update cost function metaparameters during optimization, up to once each iteration; this updating is done internally by the interfacing code (see agnostic_director).
                If scheduler is a non-callable, dynamic scheduling is disabled, and you're stuck with whatever metaparameters you started with throughout the optimization.
            paramPruner: callable or non-callable; optional. If callable, must accept paramPruner(x->array-like, its->int) where x is the current set of optimization parameters and its is the current iteration. Must return
                x_new: array-like. New set of optimization parameters.
                This function is intended to allow mutation of optimization parameters external to the minimizer during optimization, for instance purging layers which have been ablated by the algorithm. Note that if you are using bounded
                optimization and remove elements x, you must update the bounds attribute of this object to match. Otherwise, the bounds attribute will simply be concatenated at the current length of x. If non-callable, external parameter mutation is disabled
        """
        self.costFunction,self.costFunction_gradPhi,self.costFunction_gradE,self.globalBoundaries,self.transferMatrices,self.transferMatrices_gradPhi,self.scheduler,self.paramPruner=costFunction,costFunction_gradPhi,costFunction_gradE,globalBoundaries,transferMatrices,transferMatrices_gradPhi,scheduler,paramPruner
        self.simDict={'fields':[],'transferMatrices':[],'parameters':[],'previousCostFunction':np.inf,'iteration':0,}
        self.res=optimizationResults()#Instantiate the optimizationResults object, which will be returned to the user upon completion of optimization
        self.iterations=0
        self.evo=[]
        self.L=np.inf
        self.Lphys=0
        self.Lreg=0
        self.debug_verbosity=False
        agnostic_linear_adjoint.__init__(self)
        return

    #Call this method to change one or more of the interfacing functions provided when the object was instantiated. 
    #Arguments have the same form as in __init__()
    def updatePhysicsFunctions(self,costFunction='',costFunction_gradPhi='',costFunction_gradE='',transferMatrices='',transferMatrices_gradPhi='',scheduler='',paramPruner=''):
        """Optionally updates some or all functions passed to the constructor during optimization. Any argument which is not a callable is not updated. See the __init__ docstring for a description of the arguments."""
        if callable(costFunction):
            self.costFunction=costFunction
        if callable(costFunction_gradPhi):
            self.costFunction_gradPhi=costFunction_gradPhi
        if callable(costFunction_gradE):
            self.costFunction_gradE=costFunction_gradE
        if callable(transferMatrices):
            self.transferMatrices=transferMatrices
        if callable(transferMatrices_gradPhi):
            self.transferMatrices_gradPhi=transferMatrices_gradPhi
        if callable(scheduler):
            self.scheduler=secheduler
        if callable(paramPruner):
            self.paramPruner=paramPruner
        return

    #Callback is triggered after every optimization iteration
    def callback(self,x):
        """Callback function handed to minimize, called after every iteration. 
        
        Updates res attributes, and interfaces with the scheduler and any parameter mutators. It can be conveient to call callback() manually to add a set of paramters to the
        evo dictionary outside of an optimization iteration, a strategy used in agnostic_director.
            
        Args:
            x: Array-like of floats. Current value of optimization parameters, such that x[m]=phi_m
        """
        self.cb_triggered=True
        self.first_call=True
        #Add the current cost function and parameters to the evolution tracker
        self.evo.append({'costFunction':self.L,'parameters':x,'costFunction-Physical':self.Lphys,'costFunction-Regularization':self.Lreg})
        #Update the previous cost function in simDict, in case this is needed by the user to determine cost function weights
        self.simDict['previousCostFunction']=self.L
        self.simDict['iteration']+=1
        #If we are in verbose mode and it's time to print a result, do that
        if self.verbose>0 and self.iterations%self.verbose==0:
            print('Cost Function: '+str(self.L))
        self.pruned=False
        if callable(self.paramPruner) and self.during_opt:
            self._callback_prune(x)#If we're doing parameter pruning, prune some parameters
        if callable(self.scheduler) and self.during_opt:
            self._callback_ext(x)#If we're doing dynamic scheduling, decide whether to update the cost function and/or terminate now.
        self.iterations+=1
        return
    
    #Ok, so, here's a thing. Minimize can't handle a scheduler changing cost function weights on an iteration-by-iteration basis. This is because if the weight is increased after iteration n such that L_{n+1}>L_n, then the termination condition L_n-L_{n+1}<=ftol is automatically triggered. 
    #What we can do is refuse to actually terminate in this case. We go ahead and have the scheduler in the interfacing code change the weights sent to the cost function in the physics package. If the change is small enough that  L_{n+1}<L_n, all the better. If L_{n+1}>L_n, then minimize will
    #return an abnormal termination condition and stop. We ignore this, insert the current optParams as initial values in a new minimize call, and run it again with the new scheduler values. We repeat until BOTH L_{n+1}<L_n AND L_n-L_{n+1}<=ftol, which causes self.terminate to flip to True,
    #which causes the iterated minimize call loop to break once minimize finishes next (which should be immediately).
    def _callback_ext(self,x):
        """Interfaces with the scheduler"""
        self.res.nit+=1
        self.scheduler(self.Lprev,self.L,self.iterations,x,no_cb=False)#Call scheduler to allow the interfacing code to update dynamic weights internally if desired
        #If the new cost function is bigger than the old one, minimize is about to end prematurely. Set Lprev to the current L, plus 10*ftol so
        #that we don't immediately satisfy L_n-L_{n+1}<=ftol next time this function gets called, and carry on
        if self.Lprev<self.L:
            self.Lprev=self.L+10*self.ftol
        #If the new cost function is smaller than the old one and the difference is within tolerance, we're done. Terminate at the next opportunity
        elif (self.Lprev-self.L)/max(self.Lprev,self.L,1)<=self.ftol:
            self.terminate=True
            self.Lprev=self.L
        #Otherwise, update the previous cost function as the current cost function and carry on
        else:
            self.Lprev=self.L
        return
    
    #But wait, it gets worse. You think changing cost function parameters makes scipy sad, try chaging the optimziation parameters. The only way to externally change the optimization parameters then carry on the optimization is to terminate 
    #the current minimize call and restart it. So that's what we do. We collect the new optParams. If they are different, we raise a custom termOpt error to kill minimize (yes, throwing errors is literally the only way to do that), 
    #and flag that this termination is not real using self.pruned. The optimize method will see the self.pruned flag, realize that it should carry on optimizing despite minimize ending, and re-call minimize with the new parameters/bounds.
    def _callback_prune(self,x):
        """Interfaces with parameter mutator"""
        self.xp,self.bounds=self.paramPruner(x,self.iterations)
        if list(self.xp)!=list(x):
            if len(self.bounds)!=len(self.xp):
                self.bounds=self.bounds[:len(self.xp)]
            self.pruned=True
            raise termOpt#Hackiest solution in the history of hacky solutions
        return

    #Determine what kind of bounds we were given, and if we were given bounds at all
    def _processBounds(self,bounds,initialParameters):
        """Filters out different ways the user could input bounds"""
        if len(bounds)==0:
            return [],False
        elif not hasattr(bounds[0],'__iter__'):
            return [bounds,]*len(initialParameters),True
        else:
            return bounds,True

    #initialParameters: array-like, a list of initial values for the optimization parameters
    #verbose: integer, how often the code talks to you. verbose=0 means no messages before, during, or after optimization. verbose>0 means updates every *verbose* iterations, and a message when optimization begins and completes
    #ftol: real number, the relative reduction in cost function to terminate the optimization. See scipy.minimize documentation
    #gtol: real number, the relative change in gradient to terminate the optimization. See scipy.minimize documentation
    #bounds: either empty array-like, length-2 array-like, or array-like of length-2 array-likes. If empty, optimization will be unbounded and use BFGS gradient descent. If length-2, these will be the lower and upper bounds
    #        of every optimization parameter. If an array of length-2 arrays, there should be one element per optimization parameter such that bounds[m][0],bounds[m][1] are the lower,upper bounds of initialParameters[m]
    #scipy's minimize function gives me a lot less control over optimization than I would prefer, so I've had to get creative. Fair warning, this function gets a bit messy.
    def optimize(self,initialParameters,verbose=0,ftol=1E-9,gtol=1E-5,bounds=[]):
        """Primary API for inverse design. Runs the optimization given initial parameters, and returns res.
            
        Args:
            initialParameters: Array-like of floats. Starting values of optimization parameters, such that initialParameters[m]=phi_m
            verbose: Int; optional. Determines how much to print to consol. Set to 0 for silent mode. If >0, will print the current value of the cost function every **verbose** iterations. Any nonzero value will also activate some other printouts, like messages
                for when overhead tasks are done, etc.
            ftol: Float; optional. Cost function improvment tolerance for termination. See scipy.minimize documentation
            gtol: Float; optional. Gradient tolerance for termination. See scipy.minimize documentation
            bounds: Empty array-like OR two-element array-like OR array-like of two-element array-likes; optional. If empty, bounded optimization is disabled, and the BFGS algorithm is used. If a two-element array, these bounds will be used for all optimziation
                parameters. If an array-like of two-element array likes, must have one two-element bound array per optimization parameter.
                
        Returns:
            res: optimizationResults instance. Houses the results of the optimization. Intended to match format of scipy's OptimizeResult object. See optimizationResults for attributes.
        """
        self.first_call,self.terminate,self.Lprev,self.iterations,self.ftol,self.gtol,self.verbose=True,False,-np.inf,0,ftol,gtol,verbose
        del self.evo[:]
        self.bounds,use_bounds=self._processBounds(bounds,initialParameters)#Determine what kind of bounds we're using
        self._costFunc_wrapper(initialParameters)#Callback does not trigger until after the first evolution of the parameters; to ensure evo includes the initial point, simulate it first
        self.during_opt=False
        self.callback(initialParameters)#And add it to evo
        if verbose>0:
            print('Initial simulation complete.')
            print('Now beginning optimization.')
        self.iterations=0
        self.res.x=initialParameters#The initial set of parameters are, understandably, the initial set of parameters
        optimization_start=datetime.now()#Overhead is done, now the optimization clock starts ticking
        self.during_opt=True
        while not self.terminate:#While we still want to force further minimization
            self.pruned=False#Assume there is no external manipulation of params this iteration
            if self.debug_verbosity:
                print('Loop Restarted')
            self.cb_triggered=False#Assume we aren't going to trigger callback this iteration
            if not callable(self.scheduler):
                self.terminate=True#If we're not doing dynamic scheduling, only go one round of minimze
            try:#Because I need to terminate minimize if params are externally manipulated, I have to raise an error in callback if this happens. Screen for that error
                if use_bounds:#Determine whether we're using BFGS or its more restrained cousin, L-BFGS-B
                    scipy_res=minimize(self._costFunc_wrapper,self.res.x,jac=True,method='L-BFGS-B',bounds=self.bounds,callback=self.callback,options={'gtol':gtol,'ftol':ftol})
                else:
                    scipy_res=minimize(self._costFunc_wrapper,self.res.x,jac=True,method='BFGS',callback=self.callback,options={'gtol':gtol,'ftol':ftol})
                self.res.x=scipy_res.x#If minimize successfully completed, then no external param alterations occured, and our new set of params are the results of the minimization. nfev, njev, and nit are all tracked independently in callback and _costFunc_wrapper, and other scipy.res attributes will only be collected on the last minimize call, which is guaranteed to not have an external param alteration.
            except termOpt:#If the params were externally manipulated
                self.terminate=False#Restart the loop with the mutated params
                self.res.x=self.xp#Set the new params
            #print('Lprev: '+str(self.Lprev))
            #print('L: '+str(self.L))
            #print('res message: '+str(scipy_res.message))
            
            if not self.cb_triggered and not self.terminate and not self.pruned:#If the optimizer is unable to find an improvement, it can get trapped in a doom loop in which it fails to call callback to break out of this loop. This fixes that issue
                self.terminate=self.scheduler(self.Lprev,self.L,self.iterations,self.res.x,no_cb=True)
                scipy_res.message='Terminated due to meeting scheduler criteria'
                scipy_res.success=True
                scipy_res.status=0
                if self.debug_verbosity:
                    print('cb not triggered. Terminate flag: '+str(self.terminate))
        #Update attributes of optimizationResults object
        self.res.time=datetime.now()-optimization_start
        self.res.message,self.res.success,self.res.status,self.res.fun,self.res.jac=scipy_res.message,scipy_res.success,scipy_res.status,scipy_res.fun,scipy_res.jac
        self.res.evo=self.evo
        if verbose>0:
            print(self.res)
        return self.res

    def _costFunc_wrapper(self,x):
        """Wrapper for cost function. This is what minimize actually comunicates with"""
        self.res.nfev+=1
        self.res.njev+=1
        self.simulate(x)#Run the simulation to get the transfer matrices/fields
        cf_rs=self.costFunction(self.simDict)
        if hasattr(cf_rs,'__iter__'):
            self.L,self.Lphys,self.Lreg=cf_rs
        else:
            self.L=cf_rs
        dtms,nonzero_dtms=self.transferMatrices_gradPhi(self.simDict)#Get dT/dPhi
        dLdphi=self.costFunction_gradPhi(self.simDict)#Get dL/dPhi (partial derivative, not the total derivative dL/dx)
        dLde=self.costFunction_gradE(self.simDict)#Get dL/dE
        self.solve_adjoint(self.simDict['transferMatrices'],self.simDict['fields'],dLdphi,dLde,dtms,self.global_bcs,nonzero_dtms)#Get the total derivative dL/dx
        self.first_call=False#Minimize will call this up to 20 times during linesearch. Flag the first call. Not currently used for anything except debugging, but there are situations when you want the interfacing code to know when the first linesearch is happening. This is here just in case.
        return self.L,self.ala_dLdx #Return cost function and gradient

    #This is a seperate function so that the user can run a simulation and find the fields without wasting time with the derivatives if they're, say, plotting the results of a previous optimization.
    #Just instantiate this class with the correct globalBoundaries() and transferMatrices() and dummy functions for the rest, give this method the parameters, and extract the fields via agnostic_invDes.simDict['fields']
    def simulate(self,x):
        """Primary API for simulation. Runs a simulation given a set of optimization parameters.
        
            Args:
                x: Array-like of floats. Current values of optimization parameters, such that x[m]=phi_m
        """
        self.simDict['parameters']=x#Update the current optimization parameters
        del self.simDict['fields'][:]#Clear entries from previous function calls
        del self.simDict['transferMatrices'][:]
        self.global_bcs=self.globalBoundaries(self.simDict)#Get the set of global boundaries given these optimization parameters. Currently, dependence of global boundaries on parameters is not supported, but this is here in case I want to make it possible later
        self.simDict['transferMatrices']=self.transferMatrices(self.simDict)#Get the transfer matrices
        for k in range(len(self.simDict['transferMatrices'])):#Run through the simPoints
            gbs=self.global_bcs[k]
            self.simDict['fields'].append(self.solve_e(self.simDict['transferMatrices'][k],gbs[0],gbs[1],gbs[2]))#Solve for the fields at this simPoint
        return 


class optimizationResults:
    """Houses results of the optimization.
    
    This is intended to be a clone of scipy's OptimizeResults object. Since agnostic_invDes will terminate and restart minimize on a possibly regular basis, a different object had to be defined to keep track of the actual total number of 
    function evaluations, iterations, time taken, etc. Also has the added benefit of being updated dynamically as minimize progresses, so it can be accessed by interfacing code to take a peak at how things are progressing, and 
    of having the very useful evo attribute.
    
    Attributes:
        x: Array-like of floats. Current/final values of optimization parameters, such that x[m]=phi_m
        jac: Array-like of floats. Current gradient, such that jac[m]=dL/dphi_m
        time: datetime object. Time taken to perform minimization. Will be 0 until minimization completed
        evo: List of dictionaries. Tracks the evolution of the cost function and optimization parameters during every iteration of the optimization, such that evo[it]=evoDictInst where
        
            evoDictInst={'costFunction':Float, value of cost function at iteration it,
                         'parameters':Array-like of floats, the optimization parameters at iteration it,
                         'costFunction-Physical':Float, the value of the physical component of the cost function at iteration it if provided, else 0,
                         'costFunction-Regularization':Float, the value of the regularization component of the cost function at iteration it if provided, else 0}
                         
        nfev: Int. Number of function evaluations; will not in general be equal to the number of iterations as linesearch typically involves multiple function calls.
        njev: Int. Number of gradient evaluations; will almost always be equal to nfev.
        nit: Int. Number of optimization iterations.
        maxcv: Float. Maximum constraint violation.
        success: Bool, whether optimizer exited successfully.
        status: Int. Termination status of the optimizer.
        fun: Float. Final value of the cost function.
        message: Description of cause of termination.
    """
    def __init__(self): 
        """Initializes optimizationResults object"""
        self.x='Optimization Failure'
        self.jac='Optimization Failure'
        self.time=0
        self.evo=[]
        self.nfev=0
        self.njev=0
        self.nit=0
        self.maxcv=0
        self.success=False
        self.status=0
        self.fun=0
        self.message='Optimization Failure'
        return

    def prep_display(self):
        """Sets up printout of object"""
        vs=''
        xcopy=deepcopy(self.x)
        jaccopy=deepcopy(self.jac)
        if len(self.x)>10:
            x1str,x2str=str(list(xcopy[:5])),str(list(xcopy[-5:]))
            xcopy=x1str.rstrip(']')+',...,'+x2str.lstrip('[')
            jac1str,jac2str=str(list(jaccopy[:5])),str(list(jaccopy[-5:]))
            jaccopy=jac1str.rstrip(']')+',...,'+jac2str.lstrip('[')
        evo_str=str(self.evo)
        vs+=('message: '+self.message+'\n')
        vs+=('success: '+str(self.success)+'\n')
        vs+=(' status: '+str(self.status)+'\n')
        vs+=('    fun: '+str(self.fun)+'\n')
        vs+=('      x: '+str(xcopy)+'\n')
        vs+=('    nit: '+str(self.nit)+'\n')
        vs+=('    jac: '+str(jaccopy)+'\n')
        vs+=('   nfev: '+str(self.nfev)+'\n')
        vs+=('   njev: '+str(self.njev)+'\n')
        vs+=('   time: '+str(self.time)+'\n')
        vs+=('    evo: '+evo_str[:20]+'...'+evo_str[-10:])
        return vs

    def __repr__(self):
        """What happens to the object when you try and display it in consol"""
        vs=self.prep_display()
        return vs
    def __str__(self):
        """What happens to the object when you try and turn it into a string"""
        vs=self.prep_display()
        return vs
