from misc_utils import comp_utils
import numpy as np
from copy import deepcopy
from math import copysign
import cmath as cm

class optics_tmm(comp_utils):
    def __init__(self):#There are some hard-coded options. If you want to change them, edit the code
        #This caps logarithm minima to avoid underflow.
        #Also has some utility during optimization, as it prevents the code from fulfilling cost function criteria by making a single point absurdly low while ignoring the others
        self.log_cap=-400
        #The transfer matrix can be built by assuming the refractive index itself is the optParam, or the ratio of refractive indices. Not a big difference, but here if needed
        self.n_is_optParam=False
        #If true, the material ratio will be mapped from [0,1] to [0,1], but via an approximately logarithmic map (100**ratio-1)/99. Not as relevent as in acoustics, but I figured I might as well include it here.
        self.log_ratio=False
        self.zm=np.zeros((2,2),dtype=np.complex64)
        #If for some reason you don't want to normalize the incident wave or you want incoming radiation from both sides of the stack, change it here. Make sure at least one of them is a complex datatype or numpy will die later on
        self.a0,self.bN=1,0+0j
        self.kx=1
        self.avg_pol=False
        self.subLen=0.5
        return

    def get_name(self):
        return 'optics'

    '''
    Matrix Builders ##########################################################################################
    '''
    def global_bcs(self,param0,paramN,sim_params,mat1params={},mat2params={}):#Compute the global boundary matrices. This is easy.
        A=np.array(((1+0j,0),(0,0)))
        C=np.array((self.a0,self.bN))
        B=np.array(((0+0j,0),(0,1)))
        return (A,B,C)

    #Optics is you bog-standard TMM. Each matrix depends upon the current layer's x and y, and the next layer's y. It has nontrivial propagation and interface matrices.
    def left_tm(self,yp1,sim_params,matLeft={},mat1Right={},mat2Right={}):
        self.kx=np.sin(sim_params['third_vars']['incidentAngle'])#For some reason transfer matrices are almost always formulated by keeping track of propagation angle throughout the stack. You don't need to do that. You just have to know the x-direction vector, which dosen't change in an isotropic material
        if self.n_is_optParam:
            y=matLeft['refractiveIndex']
        else:
            y=1
        tm,info=self.interface_tm(0,y,yp1,sim_params,{'kx':self.kx},matLeft,matLeft,mat1Right,mat2Right)
        return tm,info
        
    def d_left_tm(self,yp1,sim_params,tracked_info,tm,matLeft={},mat1Right={},mat2Right={}):
        if self.n_is_optParam:
            y=matLeft['refractiveIndex']
        else:
            y=1
        dx,dy,dyp1=self.d_interface_tm(0,y,yp1,sim_params,tracked_info,tm,matLeft,matLeft,mat1Right,mat2Right)
        return dyp1
        
    def interface_tm(self,x,y,yp1,sim_params,tracked_info,mat1Left={},mat2Left={},mat1Right={},mat2Right={}):
        kx=tracked_info['kx']#Tracked info contains the x-direction reduced wavevector, computed in left_tm based on chosen incident angle, which tells us the angle of propagation in all succeeding layers
        if not self.n_is_optParam:
            if self.log_ratio:
                y=(100**y-1)/99
                yp1=(100**yp1-1)/99
            n1L,n2L,n1R,n2R=mat1Left['refractiveIndex'],mat2Left['refractiveIndex'],mat1Right['refractiveIndex'],mat2Right['refractiveIndex']
            nL,nR=y*n2L+(1-y)*n1L,yp1*n2R+(1-yp1)*n1R
        else:
            nL,nR=y,yp1
        epL,epR=nL**2,nR**2#Squares needed for the p-polarized R/T coefficients. Note that I'm assuming nonmagnetic materials starting here
        kzL,kzR=cm.sqrt(epL-kx**2),cm.sqrt(epR-kx**2)#Find z-direction reduced wavevectors in the left and right layers
        t,r=self.get_tr(kzL,kzR,nL,nR,sim_params['third_vars']['polarization'])
        phase=1j*kzL*(2*np.pi/sim_params['simPoint'])*x
        p=np.array(((np.exp(-1*phase),0),(0,np.exp(phase))))
        m=1/t*np.array(((1,r),(r,1)))
        return np.matmul(p,m),{'kx':kx,'p':p,'m':m}
        
    def get_tr(self,kzL,kzR,nL,nR,pol):
        epL,epR=nL**2,nR**2
        if pol=='p':
            d=epL*kzR+kzL*epR
            t,r=2*np.sqrt(epL*epR)*kzL/d,(epR*kzL-epL*kzR)/d
        else:
            d=kzR+kzL
            t,r=2*kzL/d,(kzL-kzR)/d
        return t,r
        
    def d_interface_tm(self,x,y,yp1,sim_params,tracked_info,tm,mat1Left={},mat2Left={},mat1Right={},mat2Right={}):
        kx,P,M=tracked_info['kx'],tracked_info['p'],tracked_info['m']
        if not self.n_is_optParam:
            if self.log_ratio:
                dydr=100**y/99*np.log(100)
                dyp1dr=100**yp1/99*np.log(100)
                y=(100**y-1)/99
                yp1=(100**yp1-1)/99
            else:
                dydr,dyp1dr=1,1
            n1L,n2L,n1R,n2R=mat1Left['refractiveIndex'],mat2Left['refractiveIndex'],mat1Right['refractiveIndex'],mat2Right['refractiveIndex']
            nL,nR=y*n2L+(1-y)*n1L,yp1*n2R+(1-yp1)*n1R
            dnLdy,dnRdyp1=n2L*dydr-dydr*n1L,n2R*dyp1dr-dyp1dr*n1R
        else:
            nL,nR=y,yp1
            dnLdy,dnRdyp1=1,1
        epL,epR=nL**2,nR**2
        kzL,kzR=np.sqrt(epL-kx**2),np.sqrt(epR-kx**2)
        dkzLdnL,dkzRdnR=nL/kzL,nR/kzR
        dphasedy_p=1j*2*np.pi/sim_params['simPoint']*x*dkzLdnL*dnLdy*np.exp(1j*2*np.pi/sim_params['simPoint']*x*kzL)
        dphasedy_m=-1j*2*np.pi/sim_params['simPoint']*x*dkzLdnL*dnLdy*np.exp(-1j*2*np.pi/sim_params['simPoint']*x*kzL)
        dphasedx_p=1j*2*np.pi/sim_params['simPoint']*kzL*np.exp(1j*2*np.pi/sim_params['simPoint']*x*kzL)
        dphasedx_m=-1j*2*np.pi/sim_params['simPoint']*kzL*np.exp(-1j*2*np.pi/sim_params['simPoint']*x*kzL)
        if sim_params['third_vars']['polarization']=='p':
            dr=epL*kzR+kzL*epR
            dt=2*nR*nL*kzL
            drdy=(nR**2/dr*dkzLdnL-2*nL*kzR/dr-(nR**2*kzL-nL**2*kzR)/dr**2*(nR**2*dkzLdnL+2*nL*kzR))*dnLdy
            drdyp1=(-nL**2/dr*dkzRdnR+2*nR*kzL/dr-(nR**2*kzL-nL**2*kzR)/dr**2*(nL**2*dkzRdnR+2*nR*kzL))*dnRdyp1
            dtInvdy=(nR**2/dt*dkzLdnL+2*nL*kzR/dt-(nR**2*kzL+nL**2*kzR)/(2*nR*kzL*nL**2)-(nR**2*kzL+nL**2*kzR)/(2*nR*kzL**2*nL)*dkzLdnL)*dnLdy
            dtInvdyp1=(nL**2/dt*dkzRdnR+2*nR*kzL/dt-(nR**2*kzL+nL**2*kzR)/(2*nR**2*kzL*nL))*dnRdyp1
            t,r=2*np.sqrt(epL*epR)*kzL/dr,(epR*kzL-epL*kzR)/dr
        else:
            drdy=(1/(kzL+kzR)-(kzL-kzR)/(kzL+kzR)**2)*dkzLdnL*dnLdy
            drdyp1=(-1/(kzL+kzR)-(kzL-kzR)/(kzL+kzR)**2)*dkzRdnR*dnRdyp1
            dtInvdy=(1/(2*kzL)-(kzL+kzR)/(2*kzL**2))*dkzLdnL*dnLdy
            dtInvdyp1=1/(2*kzL)*dkzRdnR*dnRdyp1
            t,r=2*kzL/(kzL+kzR),(kzL-kzR)/(kzL+kzR)
        dMdy=dtInvdy*np.array(((1,r),(r,1)))+1/t*np.array(((0,drdy),(drdy,0)))
        dMdyp1=dtInvdyp1*np.array(((1,r),(r,1)))+1/t*np.array(((0,drdyp1),(drdyp1,0)))
        dPdy=np.array(((dphasedy_m,0),(0,dphasedy_p)))
        dPdx=np.array(((dphasedx_m,0),(0,dphasedx_p)))
        return np.matmul(dPdx,M),np.matmul(dPdy,M)+np.matmul(P,dMdy),np.matmul(P,dMdyp1)
        
        
    def right_tm(self,x,y,sim_params,tracked_info,mat1Left={},mat2Left={},matRight={}):
        if self.n_is_optParam:
            yp1=matRight['refractiveIndex']
        else:
            yp1=1
        tm,info=self.interface_tm(x,y,yp1,sim_params,tracked_info,mat1Left,mat2Left,matRight,matRight)
        return tm
        
    def d_right_tm(self,x,y,sim_params,tracked_info,tm,mat1Left={},mat2Left={},matRight={}):
        if self.n_is_optParam:
            yp1=matRight['refractiveIndex']
        else:
            yp1=1
        dx,dy,dyp1=self.d_interface_tm(x,y,yp1,sim_params,tracked_info,tm,mat1Left,mat2Left,matRight,matRight)
        return dx,dy

    def tm(self,x,y,yp1,sim_params,tracked_info,mat1params={},mat2params={}):#Build a transfer matrix
        return self.interface_tm(x,y,yp1,sim_params,tracked_info,mat1params,mat2params,mat1params,mat2params)

    def dtm(self,x,y,yp1,sim_params,tracked_info,tm,mat1params={},mat2params={}):#Build the derivative of the transfer matrix
        return self.d_interface_tm(x,y,yp1,sim_params,tracked_info,tm,mat1params,mat2params,mat1params,mat2params)

    '''
    Optics indicators ###########################################################################################
    '''
    def Tdb(self,b0,aN):#Intensity transmisson coefficient, logged
        return 10*np.log10(np.abs(aN)**2*np.real(np.sqrt(self.nf**2-self.kx**2))/np.real(self.n0*np.cos(self.th0)))
    def Rdb(self,b0,aN):#Intensity reflection coefficient, logged
        return 10*np.log10(np.abs(b0)**2)
    def A(self,b0,aN):#Intensity absorption coefficient
        return 1-self.T(b0,aN)-self.R(b0,aN)
    def T(self,b0,aN):#Intensity transmission coefficient
        return np.abs(aN)**2*np.real(np.sqrt(self.nf**2-self.kx**2))/np.real(self.n0*np.cos(self.th0))
    def R(self,b0,aN):#Intensity reflection coefficient
        return np.abs(b0)**2
    def dTde(self,b0,aN):
        return np.array((0,np.real(np.sqrt(self.nf**2-self.kx**2))/np.real(self.n0*np.cos(self.th0))*2*np.conjugate(aN)))
    def dRde(self,b0,aN):
        return np.array((2*np.conjugate(b0),0))
    def dAde(self,b0,aN):
        return -1*self.dTde(b0,aN)-1*self.dRde(b0,aN)
    def dRdbde(self,b0,aN):
        return 10/(self.R(b0,aN)*np.log(10))*self.dRde(b0,aN)
    def dTdbde(self,b0,aN):
        return 10/(self.T(b0,aN)*np.log(10))*self.dTde(b0,aN)

    '''
    Cost function and partial gradients #############################################################################################
    '''
    def costFunc(self,simPoints,callPoints,third_vars,x,y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input):
        if self.avg_pol:
            return self.costFunc_avgtv(simPoints,callPoints,third_vars,x,y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input)
        else:
            return self.costFunc_navgtv(simPoints,callPoints,third_vars,x,y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input)
            
        
    #Not nearly as complicated as acoustics
    #cf_factors=[0: T linear, 1: R linear, 2: T log, 3: R log, 4: A linear, 5: [T-target,cost], 6: [R-Target, cost]]
    def costFunc_navgtv(self,simPoints,callPoints,third_vars,x,y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input):
        k=0
        tot_thickness=sum(x)
        la1,la2,la3,la4,la5,la6,la7=0,0,0,0,0,0,0
        cf_outputs=[]
        for tv in third_vars:
            for l in simPoints:
                fields,cf_factors,self.n0,self.nf,self.th0,self.kx,ci=all_fields[k],all_cf_factors[k],all_param0s[k]['refractiveIndex'],all_paramNs[k]['refractiveIndex'],tv['incidentAngle'],np.sin(tv['incidentAngle']),all_custom_input[k]
                rSub,tSub,a=0,1,0
                if len(ci.keys())>0:
                    kzL,kzR=cm.sqrt(self.nf**2-self.kx**2),cm.sqrt(ci['refractiveIndex']**2-self.kx**2)
                    a=-2*np.imag(ci['refractiveIndex'])/l*self.subLen
                    tint,rint=self.get_tr(kzL,kzR,self.nf,ci['refractiveIndex'],tv['polarization'])
                    tSub,rSub=np.abs(tint)**2*np.real(np.sqrt(ci['refractiveIndex']**2-self.kx**2))/np.real(np.sqrt(self.nf**2-self.kx**2)), np.abs(rint)**2
                tcoh,rcoh=self.T(fields[0][1],fields[-1][0]),self.R(fields[0][1],fields[-1][0])
                tl,rl=tcoh*tSub*np.exp(a)/(1-rcoh*np.exp(2*a)*rSub),rcoh+rSub*tcoh**2*np.exp(2*a)/(1-rcoh*rSub*np.exp(2*a))
                la1+=cf_factors[0]*tl #Cost of higher linear transmission; positive cf = transmission bad
                la2+=cf_factors[1]*rl #Cost of higher linear reflection; positive cf = reflection bad
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
                la5+=cf_factors[4]*self.A(fields[0][1],fields[-1][0]) #Cost of higher linear absorption; poitive cf = absorption bad
                la6+=cf_factors[5][1]*(tl-cf_factors[5][0])**2
                la7+=cf_factors[6][1]*(rl-cf_factors[6][0])**2
                k+=1
                cf_outputs.append({'tSub':tSub,'rSub':rSub,'tcoh':tcoh,'rcoh':rcoh,'tfull':tl,'rfull':rl})
        lb1=all_scheduler_factors[0]*sum(np.asarray(x)*np.asarray(y)) #Cost of total amount of material 2 used; positive cf = more material bad
        #lb2=all_scheduler_factors[1]*abs(tot_thickness-all_scheduler_factors[2]) #Cost of deviating from proscribed total footprint; positive cf = any deviation, greater or less than, bad
        lb2=all_scheduler_factors[1]*(np.arctan(all_scheduler_factors[2]*(tot_thickness-all_scheduler_factors[3]))/np.pi+0.5)*np.exp(all_scheduler_factors[4]*(tot_thickness-all_scheduler_factors[3])) #Cost of exceeding proscribed footprint. With sufficient all_scheduler_factors[2] such that the argument of arctan is large, this is approximately a heaveside centered at the desired footprint, with an exponential increase above it
        lb3=all_scheduler_factors[5]*sum([yi*(1-yi) for yi in y]) #Cost of having nondiscrete (e.g. 0 or 1) ratio; positive cf = nondiscretness bad
        lb4=all_scheduler_factors[6]*sum([abs(y[i]-y[i+1]) for i in range(len(y)-1)]) #Cost of having neighboring layers of unequal ratios; positive cf = lots of layers bad
        return la1+la2+la3+la4+la5+la6+la7+lb1+lb2+lb3+lb4,cf_outputs
        
    
    def d_costFunc_fields(self,simPoints,callPoints,third_vars,cur_x,cur_y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input,costFunc_outputs,chosen_simPoints):
        dLde=[]
        cf_needs_ds={0:self.dTde,1:self.dRde,2:self.dTdbde,3:self.dRdbde,4:self.dAde,5:self.dTde,6:self.dRde}#Determine which derivatives need to be computed based on which costFunction weights are nonzero3:None,
        cf_needs_vals={0:None,1:None,2:None,3:None,4:None,5:self.T,6:self.R}
        k=0
        tot_thickness=sum(cur_x)
        zv=[]
        for i in range(len(all_fields[0])):
            zv.append(np.zeros(2,dtype=np.complex64))
        for tv in third_vars:
            for l in simPoints:
                dlas=np.array((0+0j,0))
                fields,cf_factors,self.n0,self.nf,self.th0,self.kx,cfo=all_fields[k],all_cf_factors[k],all_param0s[k]['refractiveIndex'],all_paramNs[k]['refractiveIndex'],tv['incidentAngle'],np.sin(tv['incidentAngle']),costFunc_outputs[k]
                dLde.append(deepcopy(zv))#Each element of fields is a two-vector, giving (aj,bj), so the total number of fields is 2*len(fields)
                dt,dr=np.array(self.dTde(fields[0][1],fields[-1][0])),np.array(self.dRde(fields[0][1],fields[-1][0]))
                dtfull=dt*cfo['tSub']/(1-cfo['rcoh']*cfo['rSub'])+dr*cfo['rSub']*cfo['tcoh']*cfo['tSub']/(1-cfo['rSub']*cfo['rcoh'])**2
                drfull=dr+2*dt*cfo['tcoh']*cfo['rSub']/(1-cfo['rcoh']*cfo['rSub'])+dr*cfo['rSub']**2*cfo['tcoh']**2/(1-cfo['rSub']*cfo['rcoh'])**2
                dlas+=cf_factors[0]*dtfull
                dlas+=cf_factors[1]*drfull
                if cf_factors[2]>0:
                    dtdb=np.array(self.dTdbde(fields[0][1],fields[-1][0]))
                    if not np.isfinite(dtdb[0]):
                        dtdb[0]=0
                    if not np.isfinite(dtdb[1]):
                        dtdb[1]=0
                    dlas+=cf_factors[2]*dtdb
                if cf_factors[3]>0:
                    drdb=np.array(self.dRdbde(fields[0][1],fields[-1][0]))
                    if not np.isfinite(drdb[0]):
                        drdb[0]=0
                    if not np.isfinite(drdb[1]):
                        drdb[1]=0
                    dlas+=cf_factors[3]*drdb
                if cf_factors[4]>0:
                    dlas+=cf_factors[4]*np.array(self.dAde(fields[0][1],fields[-1][0]))
                if cf_factors[5][1]>0:
                    dlas+=cf_factors[5][1]*2*(cfo['tfull']-cf_factors[5][0])*dtfull
                if cf_factors[6][1]>0:
                    dlas+=cf_factors[6][1]*2*(cfo['rfull']-cf_factors[6][0])*drfull
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
            dLdphi[-1]+=all_scheduler_factors[0]*cur_y[i]
            #dLdphi[-1]+=all_scheduler_factors[1]*copysign(1,tot_thickness-all_scheduler_factors[2])
            dLdphi[-1]+=all_scheduler_factors[1]*all_scheduler_factors[2]/(np.pi*(1+(all_scheduler_factors[2]*(tot_thickness-all_scheduler_factors[3]))**2))*np.exp(all_scheduler_factors[4]*(tot_thickness-all_scheduler_factors[3]))+all_scheduler_factors[1]*(np.arctan(all_scheduler_factors[2]*(tot_thickness-all_scheduler_factors[3]))/np.pi+0.5)*np.exp(all_scheduler_factors[4]*(tot_thickness-all_scheduler_factors[3]))*all_scheduler_factors[4]
        #Then the d/dr
        for i in range(len(cur_y)):
            dLdphi.append(0.)
            dLdphi[-1]+=all_scheduler_factors[0]*cur_x[i]
            dLdphi[-1]+=all_scheduler_factors[5]*((1-cur_y[i])-cur_y[i])
            if i!=0:
                dLdphi[-1]+=all_scheduler_factors[6]*(-1*copysign(1,cur_y[i-1]-cur_y[i]))
            if i!=len(cur_y)-1:
                dLdphi[-1]+=all_scheduler_factors[6]*(copysign(1,cur_y[i]-cur_y[i+1]))
        return dLdphi
        
    def Twrap(self,b0,aN):
        tcoh,rcoh=self.T(b0,aN),self.R(b0,aN)
        return tcoh*self.tSub*np.exp(self.a)/(1-rcoh*self.rSub*np.exp(2*self.a))
        
    def Rwrap(self,b0,aN):
        tcoh,rcoh=self.T(b0,aN),self.R(b0,aN)
        return rcoh+self.rSub*tcoh**2*np.exp(2*self.a)/(1-rcoh*self.rSub*np.exp(2*self.a))
    '''
    Indicator computation for analysis ################################################################################ 
    '''
    def indicators(self,indicators,simPoints,third_vars,simScale,fields,mat1params,mat2params,param0s,paramNs,customInput,name):#Computes indicators for analysis
        self.sim_params={'physics':'optics','simPoint':0,'callPoint':0,'simScale':simScale}
        fSet=np.array(simPoints)*simScale#De-scale the simPoints
        indicator_units={'TInst':'','RInst':'','AInst':'','TdBInst':'dB','RdBInst':'dB'}#Units to attach to each possible indicator
        indicator_callables={'TInst':self.Twrap,'RInst':self.Rwrap,'AInst':self.A,'TdBInst':self.Tdb,'RdBInst':self.Rdb}#Dictionary of vector indicator callables
        return_vals={}
        k=0
        for tv in third_vars:#The return values will be partitioned by indicator and third variable
            for indicator in indicators:#Run through the requested indicators
                if indicator[-3:]=='Int':#Determine whether this is a scalar indicator (ends in Int for "integrated over spectrum") or a vector indicator (ends in Inst for "instantaneous"). Not currently any scalar indicators avaialble, but here if needed
                    return_vals[indicator+', '+str(tv)+', '+name]=[0,indicator_units[indicator]]
                else:
                    return_vals[indicator+', '+str(tv)+', '+name]=[[],indicator_units[indicator]]
                    return_vals[indicator+', avg, '+name]=[np.zeros(len(fSet)),indicator_units[indicator]]
            self.sim_params['third_vars']=tv
            i=0
            for f in fSet:#Run through the SI wavelengths, in m
                e=fields[k]
                self.n0,self.nf,self.th0,self.kx,ci=param0s[k]['refractiveIndex'],paramNs[k]['refractiveIndex'],tv['incidentAngle'],np.sin(tv['incidentAngle']),customInput[k]
                self.sim_params['simPoint']=simPoints[i]
                self.sim_params['callPoint']=f
                self.rSub,self.tSub,self.a=0,1,0
                if len(ci.keys())>0:
                    kzL,kzR=cm.sqrt(self.nf**2-self.kx**2),cm.sqrt(ci['refractiveIndex']**2-self.kx**2)
                    tint,rint=self.get_tr(kzL,kzR,self.nf,ci['refractiveIndex'],tv['polarization'])
                    self.tSub,self.rSub=np.abs(tint)**2*np.real(np.sqrt(ci['refractiveIndex']**2-self.kx**2))/np.real(np.sqrt(self.nf**2-self.kx**2)), np.abs(rint)**2
                    self.a=-2*np.imag(ci['refractiveIndex'])/simPoints[i]*self.subLen
                for indicator in indicators:#Craft the final return dictionary entries for vector indicators. These are updated every simPoint
                    if indicator[-4:]=='Inst': 
                        val=indicator_callables[indicator](e[0][1],e[-1][0])
                        if not np.isfinite(val):#Catch underflow values of the logarithms
                            val=-400
                        return_vals[indicator+', '+str(tv)+', '+name][0].append(val)
                        return_vals[indicator+', avg, '+name][0][i]+=val/len(third_vars)
                i+=1#i keeps track of simPoint
                k+=1#k keeps track of thrid_var/simPoint pair
        return return_vals

    def interpolate(self,sv,svd,r,mat1params,mat2params,sim_params,reverse=False,physicsPackageArg=None):
        kx=np.sin(sim_params['third_vars']['incidentAngle'])
        param0=physicsPackageArg[0](sim_params)
        n0=param0['refractiveIndex']
        physicsPackageArg=physicsPackageArg[1]
        kz0=cm.sqrt(n0**2-kx**2)
        if not self.n_is_optParam:
            if self.log_ratio:
                r=(100**r-1)/99
            n1,n2=mat1params['refractiveIndex'],mat2params['refractiveIndex']
            n=r*n2+(1-r)*n1
        else:
            n=r
        kz=np.sqrt(n**2-kx**2)
        #Propagation matrix naturally maps later state vectors to earlier vectors
        phase=1j*kz*(2*np.pi/sim_params['simPoint'])*svd*(-1)**int(not reverse)
        p=np.array(((np.exp(-1*phase),0),(0,np.exp(phase))))
        svprop=np.matmul(p,sv)
        norm=1
        if physicsPackageArg=='poyntingNorm':
            norm=np.sqrt(np.real(kx)**2+np.real(kz0)**2)
        elif physicsPackageArg=='poyntingzNorm':
            norm=np.real(kz0)
        if physicsPackageArg==None or physicsPackageArg=='intensity':
            return np.abs(svprop[0]+svprop[1])**2
        elif physicsPackageArg=='field':
            return np.real(svprop[0]+svprop[1])
        elif physicsPackageArg=='reflection':
            return np.real(svprop[1])
        elif physicsPackageArg=='reflectionB':
            return np.abs(svprop[1])**2
        elif physicsPackageArg=='transmission':
            return np.real(svprop[0])
        elif physicsPackageArg=='transmissionB':
            return np.abs(svprop[0])**2
        elif physicsPackageArg=='transmissionD':
            return np.abs(svprop[0])**2*np.real(n)**4
        elif physicsPackageArg=='reflectionD':
            return np.abs(svprop[1])**2*np.real(n)**4
        elif physicsPackageArg=='poynting' or physicsPackageArg=='poyntingNorm':
            if sim_params['third_vars']['polarization']=='s':
                return np.sqrt(np.real(np.conjugate(svprop[0]+svprop[1])*(svprop[0]+svprop[1])*kx)**2+np.real(np.conjugate(svprop[0]+svprop[1])*(svprop[0]-svprop[1])*kz)**2)/norm
            else:
                return np.sqrt(np.real(np.conjugate(svprop[0]+svprop[1])*(svprop[0]+svprop[1])*kx)**2+np.real(np.conjugate(svprop[0]-svprop[1])*(svprop[0]+svprop[1])*np.conjugate(kz))**2)/norm
        elif physicsPackageArg=='poyntingz' or physicsPackageArg=='poyntingzNorm':
            if sim_params['third_vars']['polarization']=='s':
                return np.real(np.conjugate(svprop[0]+svprop[1])*(svprop[0]-svprop[1])*kz)/norm
            else:
                return np.real(np.conjugate(svprop[0]-svprop[1])*(svprop[0]+svprop[1])*np.conjugate(kz))/norm
        elif physicsPackageArg=='poyntingzForwards':
            return np.real(np.abs(svprop[0])**2*kz)
        elif physicsPackageArg=='poyntingzBackwards':
            return np.real(np.abs(svprop[1])**2*kz)
        elif physicsPackageArg=='poyntingk' or physicsPackageArg=='poyntingkNorm':
            if sim_params['third_vars']['polarization']=='s':
                return (np.real(np.conjugate(svprop[0]+svprop[1])*(svprop[0]+svprop[1])*kx**2/n**2)+np.real(np.conjugate(svprop[0]+svprop[1])*(svprop[0]-svprop[1])*kz**2/n**2))/norm
            else:
                return (np.real(np.conjugate(svprop[0]+svprop[1])*(svprop[0]+svprop[1])*kx*np.conjugate(kx)/n**2)+np.real(np.conjugate(svprop[0]-svprop[1])*(svprop[0]+svprop[1])*kz*np.conjugate(kz)/n**2))/norm
        elif physicsPackageArg=='poyntingkB' or physicsPackageArg=='poyntingkNormB':
            if sim_params['third_vars']['polarization']=='s':
                return (np.real(np.conjugate(svprop[0]+svprop[1])*(svprop[0]+svprop[1])*kx)*np.real(kx/n**2)+np.real(np.conjugate(svprop[0]+svprop[1])*(svprop[0]-svprop[1])*kz)*np.real(kz/n**2))/norm
            else:
                return (np.real(np.conjugate(svprop[0]+svprop[1])*(svprop[0]+svprop[1])*np.conjugate(kx))*np.real(kx/n**2)+np.real(np.conjugate(svprop[0]-svprop[1])*(svprop[0]+svprop[1])*np.conjugate(kz))*np.real(kz/n**2))/norm
        elif physicsPackageArg=='poyntingzdiff':
            return abs(np.real(np.conjugate(svprop[0]+svprop[1])*(svprop[0]-svprop[1])*kz)/norm-np.real(np.conjugate(svprop[0]-svprop[1])*(svprop[0]+svprop[1])*np.conjugate(kz))/norm)
        elif physicsPackageArg=='poyntingzdiff':
            return abs(np.real(np.conjugate(svprop[0]+svprop[1])*(svprop[0]-svprop[1])*kz)/norm-np.real(np.conjugate(svprop[0]-svprop[1])*(svprop[0]+svprop[1])*np.conjugate(kz))/norm)