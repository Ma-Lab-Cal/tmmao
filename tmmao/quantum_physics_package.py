
from misc_utils import comp_utils
import numpy as np
from copy import deepcopy
from math import copysign
import cmath as cm

class quantum_tmm(comp_utils):
    def __init__(self):#There are some hard-coded options. If you want to change them, edit the code
        #This caps logarithm minima to avoid underflow.
        #Also has some utility during optimization, as it prevents the code from fulfilling cost function criteria by making a single point absurdly low while ignoring the others
        self.log_cap=-400
        self.zm=np.zeros((2,2),dtype=np.complex64)
        #If for some reason you don't want to normalize the incident wave or you want incoming radiation from both sides of the stack, change it here. Make sure at least one of them is a complex datatype or numpy will die later on
        self.a0,self.bN=1,0+0j
        self.cf_needs_ds={0:self.dTde,1:self.dRde,2:self.dTdbde,3:self.dRdbde,4:self.dTtargde,5:self.dRtargde}#Determine which derivatives need to be computed based on which costFunction weights are nonzero
        self.hbar=1
        return

    def get_name(self):
        return 'quantum'

    '''
    Matrix Builders ##########################################################################################
    '''
    def global_bcs(self,param0,paramN,sim_params,mat1params={},mat2params={}):#Compute the global boundary matrices. This is easy.
        A=np.array(((1+0j,0),(0,0)))
        C=np.array((self.a0,self.bN))
        B=np.array(((0+0j,0),(0,1)))
        return (A,B,C)

    #Quantum is a bit unique in that the wavevector can be positive, negative, or zero, each yielding different propagation matrices and every combination of those options yields a different interface matrix.
    def left_tm(self,yp1,sim_params,matLeft={},mat1Right={},mat2Right={}):
        return self.tm(0,matLeft,yp1,sim_params,{})
    def d_left_tm(self,yp1,sim_params,tracked_info,tm,matLeft={},mat1Right={},mat2Right={}):
        dx,dy,dyp1=self.dtm(0,matLeft,yp1,sim_params,tracked_info,tm)
        return dyp1
    def interface_tm(self,x,y,yp1,sim_params,tracked_info,mat1Left={},mat2Left={},mat1Right={},mat2Right={}):
        return self.tm(x,y,yp1,sim_params,tracked_info,mat1Left,mat2Left)
    def d_interface_tm(self,x,y,yp1,sim_params,tracked_info,tm,mat1Left={},mat2Left={},mat1Right={},mat2Right={}):
        return self.dtm(x,y,yp1,sim_params,tracked_info,tm,mat1Left,mat2Left)
    def right_tm(self,x,y,sim_params,tracked_info,mat1Left={},mat2Left={},matRight={}):
        tm, ti=self.tm(x,y,matRight,sim_params,tracked_info,mat1Left,mat2Left)
        return tm
    def d_right_tm(self,x,y,sim_params,tracked_info,tm,mat1Left={},mat2Left={},matRight={}):
        dx,dy,dyp1=self.dtm(x,y,matRight,sim_params,tracked_info,tm,mat1Left,mat2Left)
        return dx,dy

    def tm(self,x,y,yp1,sim_params,tracked_info,mat1params={},mat2params={}):#Build a transfer matrix
        k1,k2=cm.sqrt(2*sim_params['third_vars']['mass']*(sim_params['simPoint']-y))/self.hbar,cm.sqrt(2*sim_params['third_vars']['mass']*(sim_params['simPoint']-yp1))/self.hbar
        p=np.array(((np.exp(-1j*k1*x),0),(0,np.exp(1j*k1*x))))
        if (not abs(k1)<=1E-20) and (not abs(k2)<=1E-20):
            m=1/(2*k1)*np.array(((k1+k2,k1-k2),(k1-k2,k1+k2)))
        elif abs(k1)<=1E-20 and abs(k2)<=1E-20:
            m=np.array(((1+0j,0),(0,1)))
        elif abs(k1)<=1E-20:
            m=np.array(((1j*k2,-1j*k2),(1-1j*k2*x,1+1j*k2*x)))
        elif abs(k2)<=1E-20:
            m=1/2*np.array(((-1j/k1,1),(1j/k1,1)))
        tm=np.matmul(p,m)
        return tm,{'p':p,'m':m}

    def dtm(self,x,y,yp1,sim_params,tracked_info,tm,mat1params={},mat2params={}):#Build the derivative of the transfer matrix
        P,M=tracked_info['p'],tracked_info['m']
        k1,k2=cm.sqrt(2*sim_params['third_vars']['mass']*(sim_params['simPoint']-y))/self.hbar,cm.sqrt(2*sim_params['third_vars']['mass']*(sim_params['simPoint']-yp1))/self.hbar
        dk1dy,dk2dyp1=0,0
        if not abs(k1)<=1E-20:
            dk1dy=-0.5/self.hbar/cm.sqrt(2*sim_params['third_vars']['mass']*(sim_params['simPoint']-y))*2*sim_params['third_vars']['mass']
        if not abs(k2)<=1E-20:
            dk2dyp1=-0.5/self.hbar/cm.sqrt(2*sim_params['third_vars']['mass']*(sim_params['simPoint']-yp1))*2*sim_params['third_vars']['mass']
        dphasedk1_p,dphasedk1_m=1j*x*np.exp(1j*k1*x),-1j*x*np.exp(-1j*k1*x)
        dphasedx_p,dphasedx_m=1j*k1*np.exp(1j*k1*x),-1j*k1*np.exp(-1j*k1*x)
        dPdy=np.array(((dphasedk1_m,0),(0,dphasedk1_p)))*dk1dy
        dPdx=np.array(((dphasedx_m,0),(0,dphasedx_p)))
        if (not abs(k1)<=1E-20) and (not abs(k2)<=1E-20):
            dMdy=(-M/k1+1/(2*k1)*np.array(((1+0j,1),(1,1))))*dk1dy
            dMdyp1=(1/(2*k1)*np.array(((1+0j,-1),(-1,1))))*dk2dyp1
            dMdx=self.zm
        elif abs(k1)<=1E-20 and abs(k2)<=1E-20:
            dMdy=self.zm
            dMdyp1=self.zm
            dMdx=self.zm
        elif abs(k1)<=1E-20:
            dMdy=self.zm
            dMdyp1=np.array(((1j,-1j),(-1j*x,1j*x)))*dk2dyp1
            dMdx=np.array(((0,0),(-1j*k2,1j*k2)))
        elif abs(k2)<=1E-20:
            dMdy=0.5*np.array(((1j/k1**2,0),(-1j/k1**2,0)))*dk1dy
            dMdyp1=self.zm
            dMdx=self.zm
        dTdy=np.matmul(dPdy,M)+np.matmul(P,dMdy)
        dTdyp1=np.matmul(P,dMdyp1)
        dTdx=np.matmul(dPdx,M)+np.matmul(P,dMdx)
        return dTdx,dTdy,dTdyp1

    '''
    Quantum indicators ###########################################################################################
    '''
    def T(self,b0,aN):#Probability of transmission, linear
        return np.abs(aN)**2
    def R(self,b0,aN):#Probability of reflection, linear
        return np.abs(b0)**2
    def Tdb(self,b0,aN):#Probability of transmission, log
        return 10*np.log10(self.T(b0,aN))
    def Rdb(self,b0,aN):#Probability of reflection, log
        return 10*np.log10(self.R(b0,aN))
    def dRde(self,b0,aN):
        return np.array((2*np.conjugate(b0),0))
    def dTde(self,b0,aN):
        return np.array((0,2*np.conjugate(aN)))
    def dTdbde(self,b0,aN):
        10/(self.T(b0,aN)*np.log(10))*self.dTde(b0,aN)
    def dRdbde(self,b0,aN):
        10/(self.R(b0,aN)*np.log(10))*self.dRde(b0,aN)
    def dTtargde(self,b0,aN):
        return self.cff[1]*2*(self.T(b0,aN)-self.cff[0])*np.array(self.dTde(b0,aN))
    def dRtargde(self,b0,aN):
        return self.cff[1]*2*(self.R(b0,aN)-self.cff[0])*np.array(self.dRde(b0,aN))
        

    '''
    Cost function and partial gradients #############################################################################################
    '''
    #Not nearly as complicated as acoustics
    #cf_factors=[0: T linear, 1: R linear, 2: T log, 3: R log, 4: 4[1]*(T-4[0])**2, 5: 5[1]*(R-5[0])**2]
    def costFunc(self,simPoints,callPoints,third_vars,x,y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input):
        k=0
        tot_thickness=sum(x)
        la1,la2,la3,la4,la5,la6=0,0,0,0,0,0
        for tv in third_vars:
            for l in simPoints:
                fields,cf_factors=all_fields[k],all_cf_factors[k]
                t,r=self.T(fields[0][1],fields[-1][0]),self.R(fields[0][1],fields[-1][0])
                la1+=cf_factors[0]*t #Cost of higher linear transmission; positive cf = transmission bad
                la2+=cf_factors[1]*r #Cost of higher linear reflection; positive cf = reflection bad
                tdb=self.Tdb(fields[0][1],fields[-1][0])
                if not np.isfinite(tdb):
                    la3+=-400.*cf_factors[2]
                else:
                    la3+=cf_factors[2]*tdb #cost of higher log transmission; positive cf = transmission bad
                rdb=self.Rdb(fields[0][1],fields[-1][0])
                if not np.isfinite(rdb):
                    la4+=-400*cf_factors[3]
                else:
                    la4+=cf_factors[3]*rdb #Cost of higher log reflection; positive cf = reflection bad
                la5+=cf_factors[4][1]*(t-cf_factors[4][0])**2
                la6+=cf_factors[5][1]*(r-cf_factors[5][0])**2
                k+=1
        lb1=all_scheduler_factors[0]*sum(np.asarray(x[:-1])*np.asarray(y)) #Cost of total amount of material 2 used; positive cf = more material bad
        lb2=all_scheduler_factors[1]*abs(tot_thickness-all_scheduler_factors[2]) #Cost of deviating from proscribed total footprint; positive cf = any deviation, greater or less than, bad
        lb3=all_scheduler_factors[3]*sum([yi*(1-yi) for yi in y]) #Cost of having nondiscrete (e.g. 0 or 1) ratio; positive cf = nondiscretness bad
        lb4=all_scheduler_factors[4]*sum([abs(y[i]-y[i+1]) for i in range(len(y)-1)]) #Cost of having neighboring layers of unequal ratios; positive cf = lots of layers bad
        return la1+la2+la3+la4+la5+la6+lb1+lb2+lb3+lb4,{}


    def d_costFunc_fields(self,simPoints,callPoints,third_vars,cur_x,cur_y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input,costFunc_outputs,chosen_simPoints):
        dLde=[]
        k=0
        tot_thickness=sum(cur_x)
        zv=[]
        for i in range(len(all_fields[0])):
            zv.append(np.zeros(2,dtype=np.complex64))
        for tv in third_vars:
            for l in simPoints:
                dlas=np.array((0+0j,0))
                fields,cf_factors=all_fields[k],all_cf_factors[k]
                dLde.append(deepcopy(zv))#Each element of fields is a two-vector, giving (aj,bj), so the total number of fields is 2*len(fields)
                for j in range(len(cf_factors)):
                    if cf_factors[j]!=0 and list(cf_factors[j])!=[0,0]:
                        self.cff=cf_factors[j]
                        db0,daN=self.cf_needs_ds[j](fields[0][1],fields[-1][0])
                        if not np.isfinite(db0):
                            db0=0
                        if not np.isfinite(daN):
                            daN=0
                        if j<4:
                            dlas+=np.array((db0,daN))*cf_factors[j]
                        else:
                            dlas+=np.array((db0,daN))
                dLde[-1][0][1]+=dlas[0]
                dLde[-1][-1][0]+=dlas[1]
                k+=1
        return dLde


    #def gradLfunc_phi(self,simPoints,fields_master,lengths,ratios,param0_master,paramN_master,cf_factors_master,scheduler_factors):
    def d_costFunc_phi(self,simPoints,callPoints,third_vars,cur_x,cur_y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input,costFunc_outputs,chosen_simPoints):
        tot_thickness=sum(cur_x)
        dLdphi=[]
        #dLdphi will not be broken up by simPoint/third_var, and Lfunc does not sum the terms which depend on phi (the lbs) over simPoint/third_var, so we don't need to do so here, either.
        #First the d/dl:
        for i in range(len(cur_x)):
            dLdphi.append(0.)
            if i!=len(cur_x)-1:
                dLdphi[-1]+=all_scheduler_factors[0]*cur_y[i]
            dLdphi[-1]+=all_scheduler_factors[1]*copysign(1,tot_thickness-all_scheduler_factors[2])
        #Then the d/dr
        for i in range(len(cur_y)):
            dLdphi.append(0.)
            dLdphi[-1]+=all_scheduler_factors[0]*cur_x[i]
            dLdphi[-1]+=all_scheduler_factors[3]*((1-cur_y[i])-cur_y[i])
            if i!=0:
                dLdphi[-1]+=all_scheduler_factors[4]*(-1*copysign(1,cur_y[i-1]-cur_y[i]))
            if i!=len(cur_y)-1:
                dLdphi[-1]+=all_scheduler_factors[4]*(copysign(1,cur_y[i]-cur_y[i+1]))
        return dLdphi

    '''
    Indicator computation for analysis ################################################################################ 
    '''
    def indicators(self,indicators,simPoints,third_vars,simScale,fields,mat1params,mat2params,param0s,paramNs,customInput,name):#Computes indicators for analysis
        self.sim_params={'physics':'optics','simPoint':0,'callPoint':0,'simScale':simScale}
        fSet=np.array(simPoints)*simScale#De-scale the simPoints
        indicator_units={'TInst':'','RInst':'','TdBInst':'dB','RdBInst':'dB'}#Units to attach to each possible indicator
        indicator_callables={'TInst':self.T,'RInst':self.R,'TdBInst':self.Tdb,'RdBInst':self.Rdb}#Dictionary of vector indicator callables
        return_vals={}
        k=0
        for tv in third_vars:#The return values will be partitioned by indicator and third variable
            for indicator in indicators:#Run through the requested indicators
                if indicator[-3:]=='Int':#Determine whether this is a scalar indicator (ends in Int for "integrated over spectrum") or a vector indicator (ends in Inst for "instantaneous"). Not currently any scalar indicators avaialble, but here if needed
                    return_vals[indicator+', '+str(tv)+', '+name]=[0,indicator_units[indicator]]
                else:
                    return_vals[indicator+', '+str(tv)+', '+name]=[[],indicator_units[indicator]]
            self.sim_params['third_vars']=tv
            i=0
            for f in fSet:#Run through the SI energies, in J
                e=fields[k]
                self.sim_params['simPoint']=simPoints[i]
                self.sim_params['callPoint']=f
                for indicator in indicators:#Craft the final return dictionary entries for vector indicators. These are updated every simPoint
                    if indicator[-4:]=='Inst': 
                        val=indicator_callables[indicator](e[0][1],e[-1][0])
                        if not np.isfinite(val):#Catch underflow values of the logarithms
                            val=-400
                        return_vals[indicator+', '+str(tv)+', '+name][0].append(val)
                i+=1#i keeps track of simPoint
                k+=1#k keeps track of thrid_var/simPoint pair
        return return_vals
        
    def interpolate(self,sv,svd,r,mat1params,mat2params,sim_params,reverse=False,physicsPackageArg=None):
        k=cm.sqrt(2*sim_params['third_vars']['mass']*(sim_params['simPoint']-r))/self.hbar*(-1)**int(not reverse)
        p=np.array(((np.exp(-1j*k*svd),0),(0,np.exp(1j*k*svd))))
        svprop=np.matmul(p,sv)
        return np.abs(svprop[0]+svprop[1])**2