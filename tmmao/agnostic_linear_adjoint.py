import numpy as np

class agnostic_linear_tmm:
    """Solves an arbitrary linear tmm problem, given the list of transfer matrices.
    
    Methods:
        solve_e: solves for the field distribution
    """
    def __init__(self):
        """Initializes instance of angostic_linear_tmm"""
        return
    
    def solve_e(self,tms,A,B,C):
        """Solves for the field distribution
        
        Args:
            tms: Array-like of numpy arrays or compatible datatype. Array of transfer matrices ordered from leftmost layer's matrix to the rightmost layer's
            A: Numpy array or compatible datatype. Matrix, part of global boundary condition A.e0+B.eN=c, must be dimensionally compatible with e0.
            B: Numpy array or compatible datatype. Matrix, part of global boundary condition A.e0+B.eN=c, must be dimensionally compatible with eN.
            c: Numpy array or compatible datatype. Vector, part of global boundary condition A.e0+B.eN=c, must be dimensionally compatible e0, eN.
            
        Returns:
            e: List of numpy arrays. The list of fields, ordered from e0 to eN
            
        Raises:
            LinAlgError: Numpy raised an error when solving for eN, likely because the boundary conditions are insufficient. Ensure that A.T^{N,0}+B is a nonsingular matrix. 
        """
        e=[]
        t_full=np.identity(len(C),dtype=np.complex64)
        for t in tms:
            t_full=np.matmul(t_full,t)
        try:
            e.append(np.linalg.solve(np.matmul(A,t_full)+B,C))
        except np.linalg.LinAlgError:
            print('The boundary conditions you entered do not have a solution, which made numpy sad. The program will now terminate')
            raise np.linalg.LinAlgError
        tms.reverse()
        for t in tms:
            e.append(np.matmul(t,e[-1]))
        e.reverse()
        tms.reverse()
        return e


#Computes the total gradient of an arbitary cost function via the adjoint method.
class agnostic_linear_adjoint(agnostic_linear_tmm):
    """Solves an arbitrary linear adjoint tmm problem and computes the gradient, given the list of transfer matrices, their derivatives, and the derivatives of the cost function
    
    Subclasses:
        agnostic_linear_tmm: Solves for fields given transfer matrices. Not explicitly used in this class, but let's the user just import this class for invDes and access alt's methods too.
        
    Attributes:
        ala_eadj: List of lists of numpy arrays. The adjoint fields, such that ala_eadj[n][j]=e_j' for the nth simulation point.
        ala_dLdx: List of floats. The gradient of the cost function, such that ala_dLdx[m]=dL/dphi_m. The order of phi_m is proscribed by the dLdphi and all_dTdphi inputs
        
    Methods:
        solve_adjoint: Solves for the adjoint fields and gradient
    """
    def __init__(self):
        """Initializes instance of angostic_linear_adjoint"""
        self.ala_eadj=[]
        self.ala_dLdphi=[]
        self.ala_dLdx=[]
        self.ala_nonzero_dTdphis=[]
        self.ala_nonzero_dLdes=[]
        self.ala_nonzero_dLdphis=[]
        return

    def solve_adjoint(self,all_tms,all_fields,dLdphi,all_dLde,all_dTdphi,all_global_bc,nonzero_dTdphi=[]):
        """Solves for the adjoint fields (ala_eadj) and gradient (ala_dLdx)
        
        Args:
            all_tms: Array-like of array-likes of numpy arrays or compatible datatype. The transfer matrices, such that all_tms[n][j]=T^{j+1,j} for the nth simulation point
            all_fields: Array-like of array-likes of numpy arrays or compatible datatype. The fields, as computed by e.g. agnostic_linear_tmm, such that all_fields[n][j]=e_j for the nth simulation point
            dLdphi: Array-like of complex numbers. The partial derivatives of the cost function wrt the optimization paramters, such that dLdphi[m]=dL/dphi_m
            all_dLde: Array-like of array-likes of numpy arrays or compatible datatype. The partial derivatives of the cost function wrt the fields, such that all_dLde[n][j]=dL/de_j for the nth simulation point
            all_dTdphi: Array-like of array-likes of array-likes of numpy arrays or compatible datatype. The partial derivatives of the transfer matrices, such that all_dTdphi[n][m][j]=dT^{j+1,j}/dphi_m for the nth simulation point
            all_global_bc: Array-like of array-likes of numpy arrays or compatible datatype. The global boundary conditions, such that all_global_bc[n]=[A,B,c] for the nth simulation point, where A.e_0+B.e_N=c
            nonzero_dTdphi: Array-like of array-likes of array-likes of ints; optional. If not empty, flags which dT^{j+1,j}/dphi_m!=0 to reduce computation burden; nonzero_dTdphi[n][m]=[j_1,j_2,...] such that dT^{j_i+1,j_i}/dphi_m!=0
                for the nth simulation point.It will be assumed that dT^{j_i+1,j_i}/dphi_m==0 for any j_i not in nonzero_dTdphi[n][m] if len(nonzero_dTdphi)>0. If len(nonzero_dTdphi)==0, then it will be assumed that dT^{j_i+1,j_i}/dphi_m!=0 
                for all j_i, m, and n.
        """
        del self.ala_eadj[:]
        del self.ala_dLdx[:]
        self.ala_all_tms,self.ala_all_fields,self.ala_dLdphi,self.ala_all_dLde,self.ala_all_dTdphi,self.ala_all_global_bc=all_tms,all_fields,dLdphi,all_dLde,all_dTdphi,all_global_bc
        self._process_nonzeros(all_dTdphi,all_dLde,dLdphi,nonzero_dTdphi)
        self.ala_dim=len(self.ala_all_fields[0][0])
        self._solve_eadj()
        self._solve_dLdx()
        return

    def _process_nonzeros(self,dTdphi,dLde,dLdphi,nonzero_dTdphi):
        """Filters out default values of solve_adjoint arguments, in case the user isn't using them """
        if len(nonzero_dTdphi)==0:
            for k in range(len(dTdphi)):
                self.ala_nonzero_dTdphis.append([])
                use_dtset=dTdphi[k]
                num_dTs=len(use_dtset[0])
                for phi_set in use_dtset:
                    self.ala_nonzero_dTdphis[-1].append(set(range(num_dTs)))
        elif not hasattr(nonzero_dTdphi[0][0],'__iter__'):
            self.ala_nonzero_dTdphis=nonzero_dTdphi*len(dTdphi)
        else:
            self.ala_nonzero_dTdphis=nonzero_dTdphi
        return
        
    #dLdes is positive dL/de; manually add negative in this function, e.g. the user supplies the derivative, not the negation of the derivative
    def _solve_eadj(self):
        """Solves for adjoint fields """
        for n in range(len(self.ala_all_tms)):
            self.ala_eadj.append([])
            tmsr=reversed(list(self.ala_all_tms[n]))
            dLdesr=list(reversed(list(self.ala_all_dLde[n])))
            A,B,C=self.ala_all_global_bc[n]
            t_full=np.identity(self.ala_dim,dtype=np.complex64)
            Lsum=np.zeros(self.ala_dim,dtype=np.complex64)
            j=1
            for t in tmsr:
                t_full=np.matmul(t,t_full)
                Lsum+=np.matmul(np.transpose(t_full),-1*dLdesr[j])
                j+=1
            Lsum+=-1*dLdesr[0]
            a=np.transpose(np.matmul(A,t_full)+B)
            t_full_tp=np.transpose(t_full)
            try:
                self.ala_eadj[-1].append(np.linalg.solve(a,Lsum))
            except np.linalg.LinAlgError:
                print('The boundary conditions you entered do not have a solution, which made numpy sad. The program will now terminate')
                raise np.linalg.LinAlgError
            tms=list(self.ala_all_tms[n])
            dLdes=list(self.ala_all_dLde[n])
            self.ala_eadj[-1].append(np.matmul(np.transpose(A),self.ala_eadj[-1][-1])+dLdes[0])
            i=1
            for t in tms[:-1]:
                self.ala_eadj[-1].append(dLdes[i]+np.matmul(np.transpose(t),self.ala_eadj[-1][-1]))
                i+=1
        return

    def _solve_dLdx(self):
        """Solves for gradient """
        for m in range(len(self.ala_dLdphi)):
            self.ala_dLdx.append(0) 
        for n in range(len(self.ala_eadj)):
            eadj=self.ala_eadj[n]
            e=self.ala_all_fields[n]
            for m in range(len(self.ala_dLdphi)):
                d_sum=0
                dTdphi=self.ala_all_dTdphi[n][m]
                nonzero_ds=self.ala_nonzero_dTdphis[n][m]
                for j in nonzero_ds:
                    #The j+1 is because technically the first term of d_sum is eadj[0]*(dAdphi*e[0]+dBdphi*e[N]), but this is assumed zero and not included in dTdphi.
                    #So, dTdphi[0]=dT10/dphi, and this term of d_sum is eadj[1]*dT10/dphi*e[1]
                    d_sum+=np.matmul(eadj[j+1],np.matmul(dTdphi[j],e[j+1]))
                self.ala_dLdx[m]+=d_sum
        for m in range(len(self.ala_dLdphi)):
            self.ala_dLdx[m]=2*np.real(self.ala_dLdx[m])+self.ala_dLdphi[m]
        return
