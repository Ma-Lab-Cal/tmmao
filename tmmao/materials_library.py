
from misc_utils import comp_utils
import numpy as np
from copy import copy, deepcopy
import cmath as cm

def afunc(kpp,rho,w,a,kp):
    return np.imag(w*cm.sqrt(rho/(kp+1j*kpp)))-a

def gPermL_to_kgPerm3(val):
    return val*1E3

#l is the length of the sample
def absorption_to_attenuation(a,l):
    return -np.log(1-a)/l

cu=comp_utils()
#Finds octave band in which frequency lies. Index starts at 0.
def f2octave(f):
    return np.floor(2*np.log2(2**(1/6)/1000*f)+18)
#Finds center frequency and band given octave index
def octave2f(j,band=False):
    fcent=1000*2**((-18+j)/3)
    if band:
        return fcent, [fcent/2**(1/6),fcent*2**(1/6)]
    else:
        return fcent

def air(sim_params):#Air. An objectively overly detailed model for air.
    return_dict={}
    if sim_params['physics']=='optics':
        n=1+0.05792105/(238.0185-(sim_params['callPoint']*1E6)**-2)+0.00167917/(57.362-(sim_params['callPoint']*1E6)**-2)
        return {'refractiveIndex':n}
    T0=293.15
    T01=273.16
    Patm=101325
    k=1.42E5
    T,P,hrel=sim_params['third_vars']['temperature'],sim_params['third_vars']['pressure'],sim_params['third_vars']['humidity']
    f=sim_params['callPoint']
    w=sim_params['callPoint']*2*np.pi
    def psat(T):
        return Patm*10**(-6.8346*(T01/T)**1.261 + 4.6151)
    def h(hrel,T,P):
        return 100*hrel*(psat(T)/P)
    def FrO(hrel,T,P):
        return (P/Patm)*(24 + 4.04*10**4*h(hrel,T,P)*((0.02 + h(hrel,T,P))/(0.391 + h(hrel,T,P))))
    def FrN(hrel,T,P):
        return ((P/Patm)*(9 + (280*h(hrel,T,P))/np.exp(4.17*((T/T0)**(-3**(-1)) - 1))))/(T/T0)**2**(-1)
    def alpha(hrel,T,P,f):
        return 868.6*f**2*((1.84*(T/T0)**(1/2))/10**11 +
                           ((0.01275*(FrO(hrel,T,P)/(f**2 + FrO(hrel,T,P)**2)))/np.exp(2239.1/T) + 
                            (0.1068*(FrN(hrel,T,P)/(f**2 + FrN(hrel,T,P)**2)))/np.exp(3352/T))/(T/T0)**(5/2))
    def get_densityFull(T,P,hrel):
        p=P*Patm
        psat=6.1078*10**(7.5*T/(T+237.3))
        pvapor=psat*hrel
        pdry=p-pvapor
        rd=287.058
        rv=461.495
        return pdry/(rd*(T+T01))+pvapor/(rv*(T+T01))
    rho=get_densityFull(T,P,hrel)
    attndB=alpha(hrel,T+T01,P*Patm,f)
    attnNep=(1/(100*8.686))*attndB
    wv_real=sim_params['simPoint']*np.sqrt(rho/k)
    return_dict={'density':rho,'modulus':k,'wavevector':wv_real+1j*attnNep,'impedance':np.sqrt(rho*k)}
    return return_dict

def water(sim_params):#Water. It's water.
    logf=np.log10(sim_params['callPoint'])
    if logf<=4.72:
        dB_m=10**((7.87-1.81)/(4.72-1)*(logf-1)-7.87)
    elif logf<=5.8:
        dB_m=10**((1.81-0.765)/(5.8-4.72)*(logf-4.72)-1.81)
    else:
        dB_m=10**((1.567+0.765)/(7-5.8)*(logf-5.8)-0.765)
    data_ls=[0.0000002,0.000000225,0.00000025,0.000000275,0.0000003,0.000000325,0.00000035,0.000000375,0.0000004,0.000000425,0.00000045,0.000000475,0.0000005,0.000000525,0.00000055,0.000000575,0.0000006,0.000000625,0.00000065,0.000000675,0.0000007,0.000000725,0.00000075,0.000000775,0.0000008,0.000000825,0.00000085,0.000000875,0.0000009,0.000000925,0.00000095,0.000000975,0.000001,0.0000012,0.0000014,0.0000016,0.0000018,0.000002,0.0000022,0.0000024,0.0000026,0.00000265,0.0000027,0.00000275,0.0000028,0.00000285,0.0000029,0.00000295,0.000003,0.00000305,0.0000031,0.00000315,0.0000032,0.00000325,0.0000033,0.00000335,0.0000034,0.00000345,0.0000035,0.0000036,0.0000037,0.0000038,0.0000039,0.000004,0.0000041,0.0000042,0.0000043,0.0000044,0.0000045,0.0000046,0.0000047,0.0000048,0.0000049,0.000005,0.0000051,0.0000052,0.0000053,0.0000054,0.0000055,0.0000056,0.0000057,0.0000058,0.0000059,0.000006,0.0000061,0.0000062,0.0000063,0.0000064,0.0000065,0.0000066,0.0000067,0.0000068,0.0000069,0.000007,0.0000071,0.0000072,0.0000073,0.0000074,0.0000075,0.0000076,0.0000077,0.0000078,0.0000079,0.000008,0.0000082,0.0000084,0.0000086,0.0000088,0.000009,0.0000092,0.0000094,0.0000096,0.0000098,0.00001,0.0000105,0.000011,0.0000115,0.000012,0.0000125,0.000013,0.0000135,0.000014,0.0000145,0.000015,0.0000155,0.000016,0.0000165,0.000017,0.0000175,0.000018,0.0000185,0.000019,0.0000195,0.00002,0.000021,0.000022,0.000023,0.000024,0.000025,0.000026,0.000027,0.000028,0.000029,0.00003,0.000032,0.000034,0.000036,0.000038,0.00004,0.000042,0.000044,0.000046,0.000048,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001,0.00011,0.00012,0.00013,0.00014,0.00015,0.00016,0.00017,0.00018,0.00019,0.0002]
    data_ns=[1.40,1.37,1.36,1.35,1.35,1.35,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.32,1.32,1.32,1.31,1.31,1.30,1.28,1.24,1.22,1.19,1.16,1.14,1.15,1.20,1.29,1.37,1.43,1.47,1.48,1.48,1.47,1.45,1.43,1.42,1.41,1.40,1.39,1.37,1.36,1.36,1.35,1.35,1.34,1.34,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.32,1.32,1.31,1.31,1.30,1.29,1.28,1.26,1.25,1.27,1.32,1.36,1.36,1.35,1.34,1.33,1.33,1.32,1.32,1.32,1.31,1.31,1.31,1.31,1.30,1.30,1.30,1.30,1.29,1.29,1.29,1.28,1.28,1.27,1.26,1.26,1.25,1.24,1.23,1.22,1.19,1.15,1.13,1.11,1.12,1.15,1.18,1.21,1.24,1.27,1.30,1.33,1.35,1.38,1.40,1.42,1.44,1.46,1.48,1.48,1.49,1.50,1.51,1.52,1.53,1.54,1.55,1.55,1.55,1.55,1.55,1.54,1.53,1.52,1.52,1.52,1.53,1.54,1.56,1.59,1.70,1.82,1.89,1.92,1.96,1.97,2.00,2.04,2.06,2.07,2.08,2.09,2.11,2.12,2.13]
    data_ims=[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.01,0.02,0.06,0.12,0.19,0.27,0.30,0.27,0.24,0.19,0.14,0.09,0.06,0.04,0.03,0.02,0.01,0.01,0.01,0.00,0.00,0.00,0.00,0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.03,0.06,0.11,0.13,0.09,0.06,0.04,0.04,0.04,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.05,0.05,0.05,0.07,0.10,0.14,0.20,0.26,0.31,0.34,0.37,0.39,0.40,0.41,0.42,0.43,0.43,0.43,0.43,0.42,0.41,0.40,0.39,0.38,0.37,0.37,0.36,0.36,0.35,0.34,0.34,0.33,0.33,0.32,0.33,0.34,0.36,0.39,0.41,0.44,0.46,0.49,0.51,0.59,0.58,0.55,0.54,0.53,0.53,0.53,0.51,0.50,0.50,0.50,0.50,0.50,0.50,0.50]
    g=2.34E9
    p=1020
    j,jp1=cu.findBoundInds(data_ls,sim_params['callPoint'])
    im_wvVec=dB_m/20*np.log(10)
    kre=sim_params['callPoint']*np.sqrt(p/g)*2*np.pi
    return_dict={'density':p,'modulus':g,'wavevector':kre+1j*im_wvVec,'refractiveIndex':(data_ns[j]+data_ns[jp1])/2+1j*(data_ims[j]+data_ims[jp1])/2,'impedance':1.54E6}
    return return_dict
    
def polyurethane(sim_params):
    logf=np.log10(sim_params['callPoint'])
    if logf<=4.72:
        dB_m=10**((7.87-1.81)/(4.72-1)*(logf-1)-7.87)
    elif logf<=5.8:
        dB_m=10**((1.81-0.765)/(5.8-4.72)*(logf-4.72)-1.81)
    else:
        dB_m=10**((1.567+0.765)/(7-5.8)*(logf-5.8)-0.765)
    data_ls=[0.0000002,0.000000225,0.00000025,0.000000275,0.0000003,0.000000325,0.00000035,0.000000375,0.0000004,0.000000425,0.00000045,0.000000475,0.0000005,0.000000525,0.00000055,0.000000575,0.0000006,0.000000625,0.00000065,0.000000675,0.0000007,0.000000725,0.00000075,0.000000775,0.0000008,0.000000825,0.00000085,0.000000875,0.0000009,0.000000925,0.00000095,0.000000975,0.000001,0.0000012,0.0000014,0.0000016,0.0000018,0.000002,0.0000022,0.0000024,0.0000026,0.00000265,0.0000027,0.00000275,0.0000028,0.00000285,0.0000029,0.00000295,0.000003,0.00000305,0.0000031,0.00000315,0.0000032,0.00000325,0.0000033,0.00000335,0.0000034,0.00000345,0.0000035,0.0000036,0.0000037,0.0000038,0.0000039,0.000004,0.0000041,0.0000042,0.0000043,0.0000044,0.0000045,0.0000046,0.0000047,0.0000048,0.0000049,0.000005,0.0000051,0.0000052,0.0000053,0.0000054,0.0000055,0.0000056,0.0000057,0.0000058,0.0000059,0.000006,0.0000061,0.0000062,0.0000063,0.0000064,0.0000065,0.0000066,0.0000067,0.0000068,0.0000069,0.000007,0.0000071,0.0000072,0.0000073,0.0000074,0.0000075,0.0000076,0.0000077,0.0000078,0.0000079,0.000008,0.0000082,0.0000084,0.0000086,0.0000088,0.000009,0.0000092,0.0000094,0.0000096,0.0000098,0.00001,0.0000105,0.000011,0.0000115,0.000012,0.0000125,0.000013,0.0000135,0.000014,0.0000145,0.000015,0.0000155,0.000016,0.0000165,0.000017,0.0000175,0.000018,0.0000185,0.000019,0.0000195,0.00002,0.000021,0.000022,0.000023,0.000024,0.000025,0.000026,0.000027,0.000028,0.000029,0.00003,0.000032,0.000034,0.000036,0.000038,0.00004,0.000042,0.000044,0.000046,0.000048,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001,0.00011,0.00012,0.00013,0.00014,0.00015,0.00016,0.00017,0.00018,0.00019,0.0002]
    data_ns=[1.40,1.37,1.36,1.35,1.35,1.35,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.32,1.32,1.32,1.31,1.31,1.30,1.28,1.24,1.22,1.19,1.16,1.14,1.15,1.20,1.29,1.37,1.43,1.47,1.48,1.48,1.47,1.45,1.43,1.42,1.41,1.40,1.39,1.37,1.36,1.36,1.35,1.35,1.34,1.34,1.33,1.33,1.33,1.33,1.33,1.33,1.33,1.32,1.32,1.31,1.31,1.30,1.29,1.28,1.26,1.25,1.27,1.32,1.36,1.36,1.35,1.34,1.33,1.33,1.32,1.32,1.32,1.31,1.31,1.31,1.31,1.30,1.30,1.30,1.30,1.29,1.29,1.29,1.28,1.28,1.27,1.26,1.26,1.25,1.24,1.23,1.22,1.19,1.15,1.13,1.11,1.12,1.15,1.18,1.21,1.24,1.27,1.30,1.33,1.35,1.38,1.40,1.42,1.44,1.46,1.48,1.48,1.49,1.50,1.51,1.52,1.53,1.54,1.55,1.55,1.55,1.55,1.55,1.54,1.53,1.52,1.52,1.52,1.53,1.54,1.56,1.59,1.70,1.82,1.89,1.92,1.96,1.97,2.00,2.04,2.06,2.07,2.08,2.09,2.11,2.12,2.13]
    data_ims=[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.01,0.02,0.06,0.12,0.19,0.27,0.30,0.27,0.24,0.19,0.14,0.09,0.06,0.04,0.03,0.02,0.01,0.01,0.01,0.00,0.00,0.00,0.00,0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.03,0.06,0.11,0.13,0.09,0.06,0.04,0.04,0.04,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.05,0.05,0.05,0.07,0.10,0.14,0.20,0.26,0.31,0.34,0.37,0.39,0.40,0.41,0.42,0.43,0.43,0.43,0.43,0.42,0.41,0.40,0.39,0.38,0.37,0.37,0.36,0.36,0.35,0.34,0.34,0.33,0.33,0.32,0.33,0.34,0.36,0.39,0.41,0.44,0.46,0.49,0.51,0.59,0.58,0.55,0.54,0.53,0.53,0.53,0.51,0.50,0.50,0.50,0.50,0.50,0.50,0.50]
    p=1010
    g=2.33E9
    j,jp1=cu.findBoundInds(data_ls,sim_params['callPoint'])
    im_wvVec=dB_m/20*np.log(10)
    kre=sim_params['callPoint']*np.sqrt(p/g)*2*np.pi
    return_dict={'density':p,'modulus':g,'wavevector':kre+1j*im_wvVec,'refractiveIndex':(data_ns[j]+data_ns[jp1])/2+1j*(data_ims[j]+data_ims[jp1])/2,'impedance':1.54E6}
    return return_dict
    
    
    
def neoprene(sim_params):
    p=1310
    g=3.35E9
    dB_m=10/10000*(sim_params['callPoint'])
    kre=sim_params['callPoint']*np.sqrt(p/g)*2*np.pi
    kim=dB_m/20*np.log(10)
    return_dict={'density':p,'modulus':g,'wavevector':kre+1j*kim,'refractiveIndex':4,'impedance':np.sqrt(p*g)}
    return return_dict
    
def pvc(sim_params):
    p=1400
    g=7.93E9
    kre=sim_params['callPoint']*np.sqrt(p/g)*2*np.pi
    return_dict={'density':p,'modulus':g,'wavevector':kre,'refractiveIndex':4,'impedance':np.sqrt(p*g)}
    return return_dict

def alberich_tile(sim_params):#Anti-sonar caldding, made of porous rubber. Difficult to find since most of the stuff is classified.
    tl=4.97206703910615+(47.87709497206704-4.97206703910615)/(23.206992872278946-0.9410518204584832)*(sim_params['callPoint']/(1000)-0.9410518204584832)
    tl=tl/0.092
    a=np.log(10)/10*tl
    p=2026
    g=2.03E9
    kre=sim_params['callPoint']*np.sqrt(p/g)*2*np.pi
    return_dict={'density':p,'modulus':g,'wavevector':kre-1j*a,'refractiveIndex':1.5}
    return return_dict

def steel4340(sim_params):#High-strength steel. I don't know what they use for the pressure hulls of nuclear subs, but I assume it's 4340 steel or similar.
    if sim_params['callPoint']<=.496E-6:
        nre,nim=(2.74-1.29)/(.496E-6-.188E-6)*(sim_params['callPoint']-.188E-6)+1.29,(2.88-1.35)/(.496E-6-.188E-6)*(sim_params['callPoint']-.188E-6)+1.35
    else:
        nre,nim=(3.17-2.74)/(1.937E-6-.496E-6)*(sim_params['callPoint']-.496E-6)+2.74,(6.12-2.88)/(1.937E-6-.496E-6)*(sim_params['callPoint']-.496E-6)+2.88
    attn=1125/15*sim_params['callPoint']/(1E6)
    p=7850
    youngs=200E9
    poisson=0.29
    m_real=youngs*(1-poisson)/((1+poisson)*(1-2*poisson))
    kre=sim_params['callPoint']*np.sqrt(p/m_real)*2*np.pi
    kimag=attn*np.log(10)/20
    return {'density':p,'modulus':m_real,'wavevector':kre-1j*kimag,'refractiveIndex':nre+1j*nim}
    


def vulcanizedSiliconeRubber_k(sim_params):#Rubber, spiced with sulfur for extra zing and a lower speed of sound.
    kre=(sim_params['callPoint']*2*np.pi)/63
    g=2.9E6
    p=730
    absorps=np.array((0.1,0.2,0.15,0.4,0.15,0.53,0.31))
    freqs=np.array((500,1000,1500,2000,3000,4000,5000))
    j,jp1=cu.findBoundInds(freqs,sim_params['callPoint'])
    coeff=max(absorps[j]-(absorps[jp1]-absorps[j])/(freqs[jp1]-freqs[j])*(sim_params['callPoint']-freqs[j]),0)
    #I'm magically assuming the sample was 10cm. Idiots didn't specify thickness and the paper still somehow made it past the reviewers. Unfortunately there's surprising little data on bog-standard vulacnized silicone rubber, so I'm desperate
    kimag=-1/(2*.1)*np.log(coeff)
    return {'density':p,'modulus':g,'wavevector':kre-1j*kimag,'refractiveIndex':1.5}

def silicon(sim_params):
    lambda_um=sim_params['callPoint']*1E6
    if lambda_um<=0.295:
        nre,nim=(5.02-1.01)/(.295-.207)*(lambda_um-.207)+1.01,(5.3-3.6)/(.295-.25)*(lambda_um-.25)+3.6
    elif lambda_um<=0.344:
        nre,nim=(5.296-5.02)/(0.344-.295)*(lambda_um-.295)+5.02,(2.987-5.4)/(0.344-.295)*(lambda_um-.295)+5.4
    elif lambda_um<=0.3757:
        nre,nim=(6.709-5.296)/(0.3757-.344)*(lambda_um-.344)+5.296,(1.32-2.987)/(0.3757-.344)*(lambda_um-.344)+2.987
    else:
        nre,nim=702*np.exp(-14.327*lambda_um)+3.481,599542*np.exp(-34.459*lambda_um)
    p=2329
    g=95E9
    return {'density':p,'modulus':g,'wavevector':sim_params['simPoint']*np.sqrt(p/g),'refractiveIndex':nre+1j*nim}

def siliconDioxide(sim_params):
    l=sim_params['callPoint']
    l_in_microns=l*1E6
    n=np.sqrt(1+0.6961663*l_in_microns**2/(l_in_microns**2-0.0684043**2)+0.4079426*l_in_microns**2/(l_in_microns**2-0.1162414**2)+0.8974794*l_in_microns**2/(l_in_microns**2-9.896161**2))
    return {'refractiveIndex':n}
    
def bk7(sim_params):
    l=sim_params['callPoint']*1E6
    n=(1.485-1.52)/(2.5-.52)*(l-0.52)+1.52+1j*1E-8
    return {'refractiveIndex':n}
    
def germanium(sim_params):
    l=sim_params['callPoint']
    l_in_microns=l*1E6
    n=2.398*np.exp(-0.654912*l_in_microns-1.84389)+4.00069
    alpha_cm=10**(np.log10(0.02)+(np.log10(3)-np.log10(0.02))/(22-10)*(l_in_microns-10))
    k=l_in_microns*(alpha_cm/1E4)/(4*np.pi)
    return {'refractiveIndex':n+1j*k}
    
def zincSelenide(sim_params):
    l=sim_params['callPoint']
    l_in_microns=l*1E6
    if l_in_microns<1.08:
        n=(2.458-2.693)/(1.08-0.5)*(l_in_microns-0.5)+2.2693
    elif l_in_microns<14:
        n=2.4
    else:
        n=(2.257-2.37)/(21.75-14)*(l_in_microns-14)+2.37
    return {'refractiveIndex':n}
    
def zincSulfide(sim_params):
    l=sim_params['callPoint']
    l_in_microns=l*1E6
    if l_in_microns<29.412:
        n=-0.0066465*l_in_microns**2+0.125305*l_in_microns+2.15721
    elif l_in_microns<38.462:
        n=(6.369-0.093)/(38.462-29.412)*(l_in_microns-29.412)+0.093
    else:
        n=570*np.exp(-0.2*(l_in_microns+0.012)+2.5)+3
    if l_in_microns<27.778:
        k=(0.097-0.003)/(27.778-0.44)*(l_in_microns-0.44)+0.003
    elif l_in_microns<35.714:
        k=(5.788-0.097)/(35.714-27.778)*(l_in_microns-27.778)+0.097
    elif l_in_microns<41.667:
        k=(0.153-5.788)/(41.667-35.714)*(l_in_microns-35.714)+5.788
    else:
        k=(0.049-0.153)/(125-41.667)*(l_in_microns-41.667)+0.153
    return {'refractiveIndex':n+1j*k}
    
def potassiumBromide(sim_params):
    l=sim_params['callPoint']
    l_in_microns=l*1E6
    if l_in_microns<1:
        n=(1.5444-2.0995)/(1-0.2)*(l_in_microns-1)+1.5444
    else:
        n=-0.00019238227*l_in_microns**2+0.001751108*l_in_microns+1.535567
    return {'refractiveIndex':n}
    
def silver(sim_params):
    l=sim_params['callPoint']
    x=l*1E6
    n=1.31261-0.447013*x+0.152316*x**2-0.00227698*x**3
    k=-0.616223+7.38037*x-0.0350827*x**2-0.000676346*x**3
    return {'refractiveIndex':n+1j*k}
    
def gold(sim_params):
    l=sim_params['callPoint']
    x=l*1E6
    n=0.0134207+0.00224298*x+0.133806*x**2-0.00261924*x**3
    k=-1.12699+7.49515*x-0.0816483*x**2+0.000144921*x**3
    return {'refractiveIndex':n+1j*k}
    
def niobiumPentoxide(sim_params):
    l=sim_params['callPoint']*1E6
    if l<0.28:
        nre,nim=(3.137-3.05)/(0.28-0.25)*(l-0.28)+3.137,(0.629-1.0640)/(0.28-0.25)*(l-0.28)+0.629
    elif l<0.45:
        nre,nim=(2.452-3.137)/(0.45-0.28)*(l-0.28)+3.137,(0.00009-0.629)/(0.45-0.28)*(l-0.28)+0.629
    elif l<1:
        nre,nim=(2.258-2.452)/(1-0.45)*(l-0.45)+2.452,0
    else:
        nre,nim=(2.2265-2.258)/(2.5-1)*(l-1)+2.258,0
    return {'refractiveIndex':nre+1j*nim}
    
def zeroPotential(sim_params):
    return 0
    
def dummy_mat(sim_params):
    return {}