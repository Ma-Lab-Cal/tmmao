from misc_utils import comp_utils
import numpy as np
from copy import deepcopy
from math import copysign
import cmath as cm

class acoustics_tmm(comp_utils):
    def __init__(self):#There are some hard-coded options. If you want to change them, edit the code
        #Also has some utility during optimization, as it prevents the code from fulfilling cost function criteria by making a single point absurdly low while ignoring the others
        self.log_cap=-400
        #The transfer matrix can be built by handing it the density and complex p-wave modulus of each material, or the density and complex longitudinal wavevector
        self.use_k_directly=True
        #If true, the material ratio will be mapped from [0,1] to [0,1], but via an approximately logarithmic map (100**ratio-1)/99. This is because acoustic materials like air and cocnrete have such radically different properties, even 10% concrete
        #behaves much more like concrete than air
        self.log_ratio=True
        #Mark as true if the simPoints have already been converted to radians/scaled seconds
        self.simPoints_in_radians=False
        self.zm=np.zeros((2,2),dtype=np.complex64)
        self.z0,self.zN=1,1
        self.pressureWave=False
        self.normalizeTotal=True
        return

    def get_name(self):
        return 'acoustics'

    '''
    Matrix Builders ##########################################################################################
    '''
    def global_bcs(self,param0,paramN,sim_params,mat1params,mat2params):#Compute the global boundary matrices. This is easy.
        if self.pressureWave:
            A=np.array(((1+0j,0),(0,0)))
            B=np.array(((0+0j,0),(0,1)))
            C=np.array((1+0j,0))
        elif self.normalizeTotal:
            A=np.array(((1+0j,0),(0,0)))
            C=np.array((1,0+0j))
            B=np.array(((0+0j,0),(1,paramN['impedance'])))
        else:
            A=np.array(((-1+0j,param0['impedance']),(0,0)))
            C=np.array((2,0+0j))
            B=np.array(((0+0j,0),(1,paramN['impedance'])))
        return (A,B,C)

    #Acoustics is a bit unique in that interface matrices are all the identity, at least in pressure-velocity space along interfaces between materials of similar nature.
    #This means the transfer matrix is just the propagation matrix, which means it only depends upon the leftmost layer. This also means the incidence matrix is the identity.
    def left_tm(self,yp1,sim_params,matLeft,mat1Right,mat2Right):
            
        return np.array(((1,0),(0,1)),dtype=np.complex64),{}
    def d_left_tm(self,yp1,sim_params,tracked_info,tm,matLeft,mat1Right,mat2Right):
        return self.zm
    def interface_tm(self,x,y,yp1,sim_params,tracked_info,mat1Left,mat2Left,mat1Right,mat2Right):
        if self.log_ratio:
            y=(100**y-1)/99
            yp1=(100**yp1-1)/99
        w=sim_params['simPoint']*(2*np.pi*int(not self.simPoints_in_radians)+int(self.simPoints_in_radians))
        p1R,k1R,g1R=mat1Right['density'],mat1Right['wavevector'],mat1Right['modulus']
        p2R,k2R,g2R=mat2Right['density'],mat2Right['wavevector'],mat2Right['modulus']
        p1L,k1L,g1L=mat1Left['density'],mat1Left['wavevector'],mat1Left['modulus']
        p2L,k2L,g2L=mat2Left['density'],mat2Left['wavevector'],mat2Left['modulus']
        if self.pressureWave:
            if not self.use_k_directly:
                k1L,k2L,k1R,k2R=w*cm.sqrt(p1L/g1L),w*cm.sqrt(p2L/g2L),w*cm.sqrt(p1R/g1R),w*cm.sqrt(p2R/g2R)
            tL1R1=0.5*np.array((((1+(p1L*k1R)/(k1L*p1R))*np.exp(-1j*k1L*x),(1-(p1L*k1R)/(k1L*p1R))*np.exp(-1j*k1L*x)),((1-(p1L*k1R)/(k1L*p1R))*np.exp(1j*k1L*x),(1+(p1L*k1R)/(k1L*p1R))*np.exp(1j*k1L*x))),dtype=np.complex64)
            tL1R2=0.5*np.array((((1+(p1L*k2R)/(k1L*p2R))*np.exp(-1j*k1L*x),(1-(p1L*k2R)/(k1L*p2R))*np.exp(-1j*k1L*x)),((1-(p1L*k2R)/(k1L*p2R))*np.exp(1j*k1L*x),(1+(p1L*k2R)/(k1L*p2R))*np.exp(1j*k1L*x))),dtype=np.complex64)
            tL2R1=0.5*np.array((((1+(p2L*k1R)/(k2L*p1R))*np.exp(-1j*k2L*x),(1-(p2L*k1R)/(k2L*p1R))*np.exp(-1j*k2L*x)),((1-(p2L*k1R)/(k2L*p1R))*np.exp(1j*k2L*x),(1+(p2L*k1R)/(k2L*p1R))*np.exp(1j*k2L*x))),dtype=np.complex64)
            tL2R2=0.5*np.array((((1+(p2L*k2R)/(k2L*p2R))*np.exp(-1j*k2L*x),(1-(p2L*k2R)/(k2L*p2R))*np.exp(-1j*k2L*x)),((1-(p2L*k2R)/(k2L*p2R))*np.exp(1j*k2L*x),(1+(p2L*k2R)/(k2L*p2R))*np.exp(1j*k2L*x))),dtype=np.complex64)
            tavg=y*yp1*tL2R2+(1-y)*(1-yp1)*tL1R1+y*(1-yp1)*tL2R1+(1-y)*yp1*tL1R2
            return_matrices={'tL1R1':tL1R1,'tL1R2':tL1R2,'tL2R1':tL2R1,'tL2R2':tL2R2}
        else:
            if not self.use_k_directly:
                k1,k2=w*cm.sqrt(p1L/g1L),w*cm.sqrt(p2L/g2L)
            p11a=np.cos(k1L*x)
            p12a=-1j*p1L*w/k1L*np.sin(k1L*x)
            p21a=-1j*k1L/(p1L*w)*np.sin(k1L*x)
            p11b=np.cos(k2L*x)
            p12b=-1j*p2L*w/k2L*np.sin(k2L*x)
            p21b=-1j*k2L/(p2L*w)*np.sin(k2L*x)
            tavg=(1-y)*np.array(((p11a,p12a),(p21a,p11a)))+y*np.array(((p11b,p12b),(p21b,p11b)))
            return_matrices={'matrix_mat1':np.array(((p11a,p12a),(p21a,p11a))),'matrix_mat2':np.array(((p11b,p12b),(p21b,p11b)))}
        return tavg,return_matrices
            
        return self.tm(x,y,yp1,sim_params,tracked_info,mat1Left,mat2Left)
    def d_interface_tm(self,x,y,yp1,sim_params,tracked_info,tm,mat1Left,mat2Left,mat1Right,mat2Right):
        return self.dtm(x,y,yp1,sim_params,tracked_info,tm,mat1Left,mat2Left)
    def right_tm(self,x,y,sim_params,tracked_info,mat1Left,mat2Left,matRight):
        tm, ti=self.tm(x,y,1,sim_params,tracked_info,mat1Left,mat2Left)
        return tm
    def d_right_tm(self,x,y,sim_params,tracked_info,tm,mat1Left,mat2Left,matRight):
        return self.dtm(x,y,1,sim_params,tracked_info,tm,mat1Left,mat2Left)

    def tm(self,x,y,yp1,sim_params,tracked_info,mat1params,mat2params):#Build a transfer matrix
        if self.log_ratio:
            y=(100**y-1)/99
        w=sim_params['simPoint']*(2*np.pi*int(not self.simPoints_in_radians)+int(self.simPoints_in_radians))
        p1,k1,g1=mat1params['density'],mat1params['wavevector'],mat1params['modulus']
        p2,k2,g2=mat2params['density'],mat2params['wavevector'],mat2params['modulus']
        if self.use_k_directly:
            p11a=np.cos(k1*x)
            p12a=-1j*p1*w/k1*np.sin(k1*x)
            p21a=-1j*k1/(p1*w)*np.sin(k1*x)
            p11b=np.cos(k2*x)
            p12b=-1j*p2*w/k2*np.sin(k2*x)
            p21b=-1j*k2/(p2*w)*np.sin(k2*x)
        else:
            p11a=np.cos(w*np.sqrt(p1/g1)*x)
            p21a=-1j/np.sqrt(p1*g1)*np.sin(w*np.sqrt(p1/g1)*x)
            p12a=-1j*np.sqrt(p1*g1)*np.sin(w*np.sqrt(p1/g1)*x)
            p11b=np.cos(w*np.sqrt(p2/g2)*x)
            p21b=-1j/np.sqrt(p2*g2)*np.sin(w*np.sqrt(p2/g2)*x)
            p12b=-1j*np.sqrt(p2*g2)*np.sin(w*np.sqrt(p2/g2)*x)
        return (1-y)*np.array(((p11a,p12a),(p21a,p11a)))+y*np.array(((p11b,p12b),(p21b,p11b))),{'matrix_mat1':np.array(((p11a,p12a),(p21a,p11a))),'matrix_mat2':np.array(((p11b,p12b),(p21b,p11b)))}

    def dtm(self,x,y,yp1,sim_params,tracked_info,tm,mat1params,mat2params):#Build the derivative of the transfer matrix
        if self.log_ratio:
            drdy=100**y/99*np.log(100)
            y=(100**y-1)/99
        else:
            drdy=1
        w=sim_params['simPoint']*(2*np.pi*int(not self.simPoints_in_radians)+int(self.simPoints_in_radians))
        p1,k1,g1=mat1params['density'],mat1params['wavevector'],mat1params['modulus']
        p2,k2,g2=mat2params['density'],mat2params['wavevector'],mat2params['modulus']
        tma,tmb=tracked_info['matrix_mat1'],tracked_info['matrix_mat2']
        if self.use_k_directly:
            dPdl11a=-np.sin(k1*x)*k1
            dPdl12a=-1j*p1*w*np.cos(k1*x)
            dPdl21a=-1j*k1**2/(p1*w)*np.cos(k1*x)
            dPdl11b=-np.sin(k2*x)*k2
            dPdl12b=-1j*p2*w*np.cos(k2*x)
            dPdl21b=-1j*k2**2/(p2*w)*np.cos(k2*x)
        else:
            k1=w*np.sqrt(p1/g1)
            k2=w*np.sqrt(p2/g2)
            dPdl11a=-k1*np.sin(k1*x)
            dPdl12a=-1j*np.sqrt(p1*g1)*k1*np.cos(k1*x)
            dPdl21a=-1j/np.sqrt(p1*g1)*k1*np.cos(k1*x)
            dPdl11b=-k2*np.sin(k2*x)
            dPdl12b=-1j*np.sqrt(p2*g2)*k2*np.cos(k2*x)
            dPdl21b=-1j/np.sqrt(p2*g2)*k2*np.cos(k2*x)
        dtmdla=np.array(((dPdl11a,dPdl12a),(dPdl21a,dPdl11a)))
        dtmdlb=np.array(((dPdl11b,dPdl12b),(dPdl21b,dPdl11b)))
        return (1-y)*dtmdla+y*dtmdlb, (-1*tma+tmb)*drdy, self.zm #Return dtmdx, dtmdy, and dtmdyp1 (always zero)

    '''
    Acoustic indicators ###########################################################################################
    '''
    def Rp(self,p0,v0,pN,vN):#Presssure reflection coefficient, complex
        return (p0+self.z0*v0)/(p0-self.z0*v0)
    def Tp(self,p0,v0,pN,vN):#Pressure transmission coefficient, complex
        return pN*(self.Rp(p0,v0,pN,vN)+1)/p0
    def RI(self,p0,v0,pN,vN):#Intensity reflection coefficient, real (only active intensity, discounts reactive intensity)
        return np.abs(self.Rp(p0,v0,pN,vN))**2
    def TI(self,p0,v0,pN,vN):#Intensity transmission coefficient, real (only active intensity, discounts reactive intensity)
        return np.real(self.z0/self.zN)*np.abs(self.Tp(p0,v0,pN,vN))**2
    def AI(self,p0,v0,pN,vN):#Intensity absorption coefficient, real
        return 1-self.TI(p0,v0,pN,vN)-self.RI(p0,v0,pN,vN)
    #Transmission intensity loss, real, in dB-ref1 (10log(Intensity in/Intensity transmitted)). Differential; only valid at single frequency. Full-sprectrum loss requires integration of intensity before log
    def TLinst(self,p0,v0,pN,vN):
        return 10*np.log10(self.TI(p0,v0,pN,vN))
    #Reflection intensity loss, real, in dB-ref1 (10log(Intensity in/Intensity transmitted)). Differential; only valid at single frequency. Full-sprectrum loss requires integration of intensity before log
    def RLinst(self,p0,v0,pN,vN):
        return 10*np.log10(self.RI(p0,v0,pN,vN))
    #Absorption intensity loss, real, in dB-ref1 (10log(Intensity in/Intensity transmitted)). Differential; only valid at single frequency. Full-sprectrum loss requires integration of intensity before log
    def ALinst(self,p0,v0,pN,vN):
        return 10*np.log10(self.AI(p0,v0,pN,vN))
    def dRIde(self,p0,v0,pN,vN):
        dp0=2*np.conjugate(p0+self.z0*v0)/np.abs(p0-self.z0*v0)**2-2*np.conjugate(p0-self.z0*v0)*np.abs(p0+self.z0*v0)**2/np.abs(p0-self.z0*v0)**4
        dv0=self.z0*2*np.conjugate(p0+self.z0*v0)/np.abs(p0-self.z0*v0)**2+self.z0*2*np.conjugate(p0-self.z0*v0)*np.abs(p0+self.z0*v0)**2/np.abs(p0-self.z0*v0)**4
        return np.array((dp0,dv0,0,0))
    def dTIde(self,p0,v0,pN,vN):
        prefac=np.real(self.z0/self.zN)
        dp0=prefac*(-8)*np.abs(pN)**2*np.conjugate(p0-self.z0*v0)/np.abs(p0-self.z0*v0)**4
        dv0=prefac*(8*self.z0)*np.abs(pN)**2*np.conjugate(p0-self.z0*v0)/np.abs(p0-self.z0*v0)**4
        dpN=prefac*(8)*np.conjugate(pN)/np.abs(p0-self.z0*v0)**2
        return np.array((dp0,dv0,dpN,0))
    def dAIde(self,p0,v0,pN,vN):
        return -1*self.dTIde(p0,v0,pN,vN)-1*self.dRIde(p0,v0,pN,vN)
    def dRLinstde(self,p0,v0,pN,vN):
        return 10/(self.RI(p0,v0,pN,vN)*np.log(10))*self.dRIde(p0,v0,pN,vN)
    def dTLinstde(self,p0,v0,pN,vN):
        return 10/(self.TI(p0,v0,pN,vN)*np.log(10))*self.dTIde(p0,v0,pN,vN)
    def dALinstde(self,p0,v0,pN,vN):
        return 10/(self.AI(p0,v0,pN,vN)*np.log(10))*self.dAIde(p0,v0,pN,vN)

    '''
    Cost function and partial gradients #############################################################################################
    '''
    #cf_factors=[0: RI linear, 1: TI linear, 2: AI linear, 3: RL log, 
    #            4: TL log, 5: AL log, 6: Iref linear, 7: Itrans linear, 
    #            8: log(sum(int(RI*8[0])))*8[1], 9: log(sum(int(TI*9[0])))*9[1],
    #            10: log(sum(int(AI*10[0])))*10[1], 11: log(sum(int(Iin))/sum(int(Iref))), 
    #            12: log(sum(int(Iin))/sum(int(Itrans))), 13: log(sum(int(Iin))/sum(int(Iref)+int(Itrans)))]
    #All >=11 applied only once, to final log/sum, rather than individually at each frequency/thrid_var combo
    #Integrals include cf_factors applied to kernal elements
    def costFunc(self,simPoints,callPoints,third_vars,x,y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input):
        num_simPoints=len(simPoints)
        tot_thickness=sum(x)
        la1,la2,la3,la4,la5,la6,la7,la8=0,0,0,0,0,0,0,0
        RIint,TIint,AIint,IinIntsCustom,IrefIntsCustom,ItransIntsCustom=0,0,0,0,0,0
        dws=[]
        lb1,lb2,lb3,lb4=0,0,0,0
        for n in range(len(all_fields)):
            s=n%num_simPoints#s iterates through simPoints only, while n iterates through thrid_var/simPoint pairs
            fields,cf_factors,self.z0,self.zN,custom_Iin_dB=all_fields[n],all_cf_factors[n],all_param0s[n]['impedance'],all_paramNs[n]['impedance'],all_custom_input[n]['custom_Iin_dB']
            IinCustom=10**(custom_Iin_dB/10)
            df=(callPoints[min(s+1,num_simPoints-1)]-callPoints[max(0,s)])#Current integration interval, in Hz
            dws.append(df)
            p0,v0,pN,vN=fields[0][0],fields[0][1],fields[-1][0],fields[-1][1]
            ri,ti,ai=float(self.RI(p0,v0,pN,vN)),float(self.TI(p0,v0,pN,vN)),float(self.AI(p0,v0,pN,vN))
            la1+=self.mul(cf_factors[0],ri)#Cost of higher linear intensity reflection; positive cf = reflection bad
            la2+=self.mul(cf_factors[1],ti)#Cost of higher linear intensity transmission; positive cf = transmission bad
            la3+=self.mul(cf_factors[2],ai)#Cost of higher linear intensity absorption; positive cf = absorption bad
            la4+=self.mul(cf_factors[3],float(max(self.RLinst(p0,v0,pN,vN),self.log_cap))) #Cost of higher single-frequency reflection loss; positive cf = reflection good
            la5+=self.mul(cf_factors[4],float(max(self.TLinst(p0,v0,pN,vN),self.log_cap))) #Cost of higher single-frequency transmission loss; positive cf = transmission good
            la6+=self.mul(cf_factors[5],float(max(self.ALinst(p0,v0,pN,vN),self.log_cap))) #Cost of higher single-frequency absorption loss; positive cf = absorption good
            la7+=self.mul(cf_factors[6],ri*IinCustom) #Cost of higher reflected intensity; positive cf = reflection bad
            la8+=self.mul(cf_factors[7],ti*IinCustom) #Cost of higher transmitted intensity; positive cf = reflection bad
            #"Integration" is a left-handed Riemann sum
            RIint+=df*cf_factors[8][0]*ri
            TIint+=df*cf_factors[9][0]*ti
            AIint+=df*cf_factors[10][0]*ai
            IinIntsCustom+=IinCustom*df
            IrefIntsCustom+=IinCustom*df*ri
            ItransIntsCustom+=IinCustom*df*ti
            #print('n: '+str(n)+', relevent_fields: '+str([fields[0][0],fields[0][-1],fields[-1][0],fields[-1][-1]])+', ti: '+str(ti))
        la9=max(self.mul(10*self.log10(RIint),cf_factors[8][1]),-400)
        la10=max(self.mul(10*self.log10(TIint),cf_factors[9][1]),-400)
        la11=max(self.mul(10*self.log10(AIint),cf_factors[10][1]),-400)
        la12=max(self.mul(10*self.log10(self.div(IinIntsCustom,IrefIntsCustom)),cf_factors[11]),-400)
        la13=max(self.mul(10*self.log10(self.div(IinIntsCustom,ItransIntsCustom)),cf_factors[12]),-400)
        la14=max(self.mul(10*self.log10(self.div(IinIntsCustom,ItransIntsCustom+IrefIntsCustom)),cf_factors[13]),-400)
        lb1=all_scheduler_factors[0]*sum(np.asarray(x)*np.asarray(x)) #Cost of total amount of material 2 used; positive cf = more material bad
        #lb2=all_scheduler_factors[1]*abs(tot_thickness-all_scheduler_factors[2]) #Cost of deviating from proscribed total footprint; positive cf = any deviation, greater or less than, bad
        lb2=all_scheduler_factors[1]*(np.arctan(all_scheduler_factors[2]*(tot_thickness-all_scheduler_factors[3]))/np.pi+0.5)*np.exp(all_scheduler_factors[4]*(tot_thickness-all_scheduler_factors[3])) #Cost of exceeding proscribed footprint. With sufficient all_scheduler_factors[2] such that the argument of arctan is large, this is approximately a heaveside centered at the desired footprint, with an exponential increase above it
        lb3=all_scheduler_factors[5]*sum([r*(1-r) for r in y]) #Cost of having nondiscrete (e.g. 0 or 1) ratio; positive cf = nondiscretness bad
        lb4=all_scheduler_factors[6]*sum([abs(y[i]-y[i+1]) for i in range(len(y)-1)]) #Cost of having neighboring layers of unequal ratios; positive cf = lots of layers bad
        Lphys=la1+la2+la3+la4+la5+la6+la7+la8+la9+la10+la11+la12+la13+la14
        Lreg=lb1+lb2+lb3+lb4
        extra={'dw':dws,'RIint':RIint,'TIint':TIint,'AIint':AIint,'IrefIntsCustom':IrefIntsCustom,'ItransIntsCustom':ItransIntsCustom}
        return Lphys+Lreg,extra,Lphys,Lreg


    #Slight problem in that computing everthing takes about 5 seconds, which is a long time if it has to be done ten thousand times. Much of the complexity here is ensuring the absolute minimum necessary computation must ever be done
    def d_costFunc_fields(self,simPoints,callPoints,third_vars,cur_x,cur_y,all_mat1params,all_mat2params,all_param0s,all_paramNs,all_fields,all_tms,all_global_bcs,all_cf_factors,all_scheduler_factors,all_custom_input,costFunc_outputs,chosen_simPoints):
        cf_needs_ds={0:[self.dRIde],1:[self.dTIde],2:[self.dAIde],3:[self.dRLinstde],4:[self.dTLinstde],5:[self.dALinstde],6:[self.dRIde],7:[self.dTIde],
                     8:[self.dRIde],9:[self.dTIde],10:[self.dAIde],11:[self.dRIde],12:[self.dTIde],13:[self.dRIde,self.dTIde]}#Determine which derivatives need to be computed based on which costFunction weights are nonzero
        derivs_dict={}
        dLde=[]
        num_simPoints=len(simPoints)
        #dLde will be broken up by simPoint/third_var for ease of adjoint calculation later. 
        for n in range(len(all_fields)):
            s=n%num_simPoints
            dlas=np.array((0,0,0,0),dtype=np.complex64)#dlas is dL/dp0, dL/dv0, dL/dpN, dL/dvN. This assumes the cost function only depends upon the first and last state vectors
            if s in chosen_simPoints:#If this is a chosen simPoint, find derivatives. Otherwise, dlas will be kept as all 0's
                nz_cf=tuple([i for i in range(len(all_cf_factors[n])) if all_cf_factors[n][i]!=0 and all_cf_factors[n][i]!=[0,0] and all_cf_factors[n][i]!=[0,0,0] and all_cf_factors[n][i]!=[0,0,0,0]])#Figure out which cost function weights are non-zero
                needed_derivs=set()
                for cf_ind in nz_cf:
                    for d in cf_needs_ds[cf_ind]:
                        needed_derivs.add(d)
                dLde.append([])
                #Grab the appropriate parameters for this particular simPoint/third_var,including the dw found in costFunc
                fields,cf_factors,constr_factors,self.z0,self.zN,dw,custom_Iin_dB=all_fields[n],all_cf_factors[n],all_scheduler_factors,all_param0s[n]['impedance'],all_paramNs[n]['impedance'],costFunc_outputs['dw'][n],all_custom_input[n]['custom_Iin_dB']
                IinCustom=10**(custom_Iin_dB/10)
                p0,v0,pN,vN=fields[0][0],fields[0][1],fields[-1][0],fields[-1][1]
                derivs_dict.clear()
                
                for needed_deriv in needed_derivs:#Compute the needed derivatives
                    nd=list(needed_deriv(p0,v0,pN,vN))
                    for j in range(len(nd)):
                        if not np.isfinite(nd[j]):
                            nd[j]=0
                    derivs_dict[needed_deriv.__name__]=np.array(nd)
                if cf_factors[0]!=0:#Apply the appropriate multiplicative factors to the derivatives and add them to dlas
                    dlas+=self.mul(derivs_dict['dRIde'],cf_factors[0])
                if cf_factors[1]!=0:
                    dlas+=self.mul(derivs_dict['dTIdec'],cf_factors[1])
                if cf_factors[2]!=0:
                    dlas+=self.mul(derivs_dict['dAIdec'],cf_factors[2])
                if cf_factors[3]!=0:
                    dlas+=self.mul(derivs_dict['dRLinstde'],cf_factors[3])
                if cf_factors[4]!=0:
                    dlas+=self.mul(derivs_dict['dTLinstde'],cf_factors[4])
                if cf_factors[5]!=0:
                    dlas+=self.mul(derivs_dict['dALinstde'],cf_factors[5])
                if cf_factors[6]!=0:
                    dlas+=self.mul(derivs_dict['dRIde'],IinCustom*cf_factors[6])
                if cf_factors[7]!=0:
                    dlas+=self.mul(derivs_dict['dTIde'],IinCustom*cf_factors[7])
                if cf_factors[8][-1]!=0:
                    dlas+=self.mul(self.mul(self.mul(derivs_dict['dRIde'],cf_factors[8][0]),self.div(10,costFunc_outputs['RIint']*np.log(10))),cf_factors[8][1]*dw)
                if cf_factors[9][-1]!=0:
                    dlas+=self.mul(self.mul(self.mul(derivs_dict['dTIde'],cf_factors[9][0]),self.div(10,costFunc_outputs['TIint']*np.log(10))),cf_factors[9][1]*dw)
                if cf_factors[10][-1]!=0:
                    dlas+=self.mul(self.mul(self.mul(derivs_dict['dAIde'],cf_factors[10][0]),self.div(10,costFunc_outputs['AIint']*np.log(10))),cf_factors[10][1]*dw)
                if cf_factors[11]!=0:
                    dIref=self.mul(derivs_dict['dRIde'],10**(custom_Iin_dB/10)*dw)
                    dlas+=self.div(self.mul((-dIref)*10/(np.log(10)),cf_factors[11]),costFunc_outputs['IrefIntsCustom'])
                if cf_factors[12]!=0:
                    dItrans=self.mul(derivs_dict['dTIde'],10**(custom_Iin_dB/10)*dw)
                    dlas+=self.div(self.mul((-dItrans)*10/(np.log(10)),cf_factors[12]),costFunc_outputs['ItransIntsCustom'])
                if cf_factors[13]!=0:
                    dItrans=self.mul(derivs_dict['dTIde'],10**(custom_Iin_dB/10)*dw)
                    dIref=self.mul(derivs_dict['dRIde'],10**(custom_Iin_dB/10)*dw)
                    dlas+=self.div(self.mul((-dItrans)*10/(np.log(10)),cf_factors[13]),costFunc_outputs['ItransIntsCustom'])+self.div(self.mul((-dIref)*10/(np.log(10)),cf_factors[13]),costFunc_outputs['IrefIntsCustom'])
            #Insert zeros for all dL/de except e=p0,v0,pN,vN
            for i in range(len(fields)):
                if i==0:
                    dLde[-1].append(np.array((dlas[0],dlas[1])))
                elif i==len(fields)-1:
                    dLde[-1].append(np.array((dlas[2],dlas[3])))
                else:
                    dLde[-1].append(np.array((0,0+0j)))
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
            #if abs(tot_thickness-all_scheduler_factors[2])>=1E-12:
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

    '''
    Indicator computation for analysis ################################################################################ 
    '''
    #dBA weighting, for sensitivity-adjusted acoustic power. f in Hz
    def a_weight(self,f):
        return 12194**2*f**4/((f**2+20.6**2)*np.sqrt((f**2+107.7**2)*(f**2+737.9**2))*(f**2+12194**2))
    
    def indicators(self,indicators,simPoints,third_vars,simScale,fields,mat1params,mat2params,param0s,paramNs,customInput,name):#Computes indicators for analysis
        self.sim_params={'physics':'acoustics','simPoint':0,'callPoint':0,'simScale':simScale}
        add_A_weight=False#Hard-coded option to apply A-weight to results
        dB_ref=1E-12#Hard-coded reference for dB intensity values. 1 pW/m^2 is the standard
        fSet=np.array(simPoints)*simScale#De-scale the simPoints
        indicator_units={'RIInst':'','TIInst':'','AIInst':'','RLInst':'dB','TLInst':'dB','ALInst':'dB','RIInt':'dB','TIInt':'dB','AIInt':'dB','IinInt':'dB','IrefInt':'dB','ItransInt':'dB','RLInt':'dB','TLInt':'dB','ALInt':'dB'}#Units to attach to each possible indicator
        indicator_callables={'RIInst':self.RI,'TIInst':self.TI,'AIInst':self.AI,'RLInst':self.RLinst,'TLInst':self.TLinst,'ALInst':self.ALinst}#Dictionary of vector indicator callables
        return_vals={}
        k=0
        for tv in third_vars:#The return values will be partitioned by indicator and third variable
            for indicator in indicators:#Run through the requested indicators
                if indicator[-3:]=='Int':#Determine whether this is a scalar indicator (ends in Int for "integrated over spectrum") or a vector indicator (ends in Inst for "instantaneous")
                    return_vals[indicator+', '+str(tv)+', '+name]=[0,indicator_units[indicator]]
                else:
                    return_vals[indicator+', '+str(tv)+', '+name]=[[],indicator_units[indicator]]
            self.sim_params['third_vars']=tv
            intensity_scale=simScale**3#Intesnsity is kg/s^3. We will return intensity in SI units. simScale is the scaling for Hz=1/s, so the intensity scale is simScale^3
            IinSum,ItransSum,IrefSum,IaSum,RLSum,TLSum,ALSum=0,0,0,0,0,0,0
            i=0
            dfs=[]
            for f in fSet:#Run through the SI frequencies, in Hz
                e=fields[k]
                self.z0=param0s[k]['impedance']
                self.zN=paramNs[k]['impedance']
                if add_A_weight:#Compute the A-weight at this frequency if we're using A-weighting
                    weight=self.a_weight(fSet[i])
                else:
                    weight=1
                df=((fSet[min(i+1,len(fSet)-1)]-fSet[max(i-1,0)])/2)#Compute the frequency interval for integrating
                dfs.append(df)
                self.sim_params['simPoint']=simPoints[i]
                self.sim_params['callPoint']=f
                ti=self.TI(e[0][0],e[0][1],e[-1][0],e[-1][1])#Compute intensity transmission coefficient
                ri=self.RI(e[0][0],e[0][1],e[-1][0],e[-1][1])#Compute intensity reflection coefficient
                ai=self.AI(e[0][0],e[0][1],e[-1][0],e[-1][1])#Compute intensity absorption coefficient
                if len(customInput)!=0:#If we've been given a custom incident intensity spectrum, use it for the incident intensity. The custom spectrum will be assumed to be in SI, with reference intensity being dB_ref
                    Iin=customInput[k]['custom_Iin_dB']
                    Iin=10**(Iin/10)*dB_ref
                    Itrans=Iin*ti
                    Iref=Iin*ri
                else:#Otherwise, use the simulated incident intensity
                    Iin=self.IinInst(e[0][0],e[0][1],e[-1][0],e[-1][1])
                    Itrans=self.ItransInst(e[0][0],e[0][1],e[-1][0],e[-1][1])
                    Iref=self.IrefInst(e[0][0],e[0][1],e[-1][0],e[-1][1])
                
                RLSum+=ri*df#"Integrate" via a midpoint reimann sum
                TLSum+=ti*df
                ALSum+=ai*df
                IinSum+=Iin*intensity_scale*weight*df
                ItransSum+=Itrans*intensity_scale*weight*df
                IrefSum+=Iref*intensity_scale*weight*df
                for indicator in indicators:#Craft the final return dictionary entries for vector indicators. These are updated every simPoint
                    if indicator[-4:]=='Inst': 
                        val=indicator_callables[indicator](e[0][0],e[0][1],e[-1][0],e[-1][1])
                        if not np.isfinite(val):#Catch underflow values of the logarithms
                            val=-400
                        return_vals[indicator+', '+str(tv)+', '+name][0].append(val)
                i+=1#i keeps track of simPoint
                k+=1#k keeps track of thrid_var/simPoint pair
            tot_f=fSet[-1]-fSet[0]
            integ_vals={'RIInt':-10*np.log10(IinSum/IrefSum),'TIInt':-10*np.log10(IinSum/ItransSum),'AIInt':-10*np.log10(IinSum/(ItransSum+IrefSum)),'IinInt':10*np.log10(IinSum/(dB_ref)),'IrefInt':10*np.log10(IrefSum/(dB_ref)),'ItransInt':10*np.log10(ItransSum/(dB_ref)),
                        'RLInt':-10*np.log10(RLSum/tot_f),'TLInt':-10*np.log10(TLSum/tot_f),'ALInt':-10*np.log10(ALSum/tot_f)}#Compute the relevent integrated values once all frequencies have been integrated over
            for indicator in indicators:#Add them to the results dicitonary
                if indicator[-3:]=='Int':
                    return_vals[indicator+', '+str(tv)+', '+name][0]+=integ_vals[indicator]
        return return_vals

    def interpolate(self,sv,svd,r,mat1params,mat2params,sim_params,reverse=False,physicsPackageArg=[1,None]):
        scale=physicsPackageArg[0]
        physicsPackageArg=physicsPackageArg[1]
        if self.log_ratio:
            r=(100**r-1)/99
        w=sim_params['simPoint']*(2*np.pi*int(not self.simPoints_in_radians)+int(self.simPoints_in_radians))
        p1,k1,g1=mat1params['density'],mat1params['wavevector'],mat1params['modulus']
        p2,k2,g2=mat2params['density'],mat2params['wavevector'],mat2params['modulus']
        x=svd
        if self.use_k_directly:
            p11a=np.cos(k1*x)
            p12a=-1j*p1*w/k1*np.sin(k1*x)
            p21a=-1j*k1/(p1*w)*np.sin(k1*x)
            p11b=np.cos(k2*x)
            p12b=-1j*p2*w/k2*np.sin(k2*x)
            p21b=-1j*k2/(p2*w)*np.sin(k2*x)
        else:
            p11a=np.cos(w*np.sqrt(p1/g1)*x)
            p21a=-1j/np.sqrt(p1*g1)*np.sin(w*np.sqrt(p1/g1)*x)
            p12a=-1j*np.sqrt(p1*g1)*np.sin(w*np.sqrt(p1/g1)*x)
            p11b=np.cos(w*np.sqrt(p2/g2)*x)
            p21b=-1j/np.sqrt(p2*g2)*np.sin(w*np.sqrt(p2/g2)*x)
            p12b=-1j*np.sqrt(p2*g2)*np.sin(w*np.sqrt(p2/g2)*x)
        p=(1-r)*np.array(((p11a,p12a),(p21a,p11a)))+r*np.array(((p11b,p12b),(p21b,p11b)))
        if not reverse:
            p=np.linalg.inv(p)#p naturally maps later svs to earlier ones. Unless we're reverse-propagating, we want to invert that to map to later svs
        svprop=np.matmul(p,sv)
        impedance=(1-r)*np.sqrt(g1/p1)+r*np.sqrt(g2/p2)
        if physicsPackageArg==None or physicsPackageArg=='activeIntensity':
            return 0.5*np.real(svprop[0]*np.conjugate(svprop[1]))*scale
        elif physicsPackageArg=='reactiveIntensity':
            return 0.5*np.imag(svprop[0]*np.conjugate(svprop[1]))*scale
        elif physicsPackageArg=='activeIntensityB':
            return abs(0.5*np.real(svprop[0]*np.conjugate(svprop[1])))**2*scale
        elif physicsPackageArg=='reactiveIntensityB':
            return abs(0.5*np.imag(svprop[0]*np.conjugate(svprop[1])))**2*scale
        elif physicsPackageArg=='forwardsIntensity':
            return 1/(4)*np.abs(svprop[0]+impedance*svprop[1])**2*scale
        elif physicsPackageArg=='backwardsIntensity':
            return 1/(4)*np.abs(svprop[0]-impedance*svprop[1])**2*scale
        elif physicsPackageArg=='pressure':
            return np.real(svprop[0])*scale
        elif physicsPackageArg=='velocity':
            return np.real(svprop[1])*scale
        elif physicsPackageArg=='pressureB':
            return np.abs(svprop[0])**2*scale
        elif physicsPackageArg=='velocityB':
            return np.abs(svprop[1])**2*scale
        elif physicsPackageArg=='absorption':
            avgAbs=(1-r)*(np.real(k1)*np.imag(k1)/(p1*w))+r*(np.real(k2)*np.imag(k2)/(p2*w))
            return abs(avgAbs)*np.abs(svprop[0])**2*scale
            