import numpy as np
from uproot_methods.classes.TLorentzVector import TLorentzVectorArray as vlv
from coffea.analysis_objects import JaggedCandidateArray as Jca
import uproot_methods
import math
from awkward import JaggedArray
import awkward
from numba import jit, njit
from numba.experimental import jitclass
from numba import float32, float64, int64
import numba
import pdb, traceback, sys
from functools import partial
try: from scipy.optimize import leastsq
except: 
    leastsq=None
    print("NO least squares")

@njit(cache=True)
def UnitCircle():
    '''Unit circle in extended representation.'''
    diag = np.array([1.,1.,-1.])
    return np.diag(diag)


@njit(cache=True)
def cofactor(A, ij):
    '''Cofactor[i,j] of 3x3 matrix A.'''
    i,j =ij
    iidx, jidx = [0,1,2], [0,1,2]
    iidx.pop(i)
    jidx.pop(j)
    B = A[np.array(iidx), :]
    a = B[:, np.array(jidx)]
    return (-1)**(i+j) * (a[0,0]*a[1,1] - a[1,0]*a[0,1])


@njit(cache=True)
def R(axis, angle):
    '''Rotation matrix about x,y, or z axis (0,1, or 2, respectively.)'''
    c,s = math.cos(angle),math.sin(angle)
    R = c * np.eye(3)
    for i in [-1,0,1] : R[ (axis-i)%3, (axis+i)%3 ] = i*s + (1 - i*i)
    return R


@njit(cache=True)
def Derivative():
    '''Matrix to differentiate [cos(t),sin(t),1] and R(2,_).'''
    return R(2, math.pi / 2).dot(np.diag([1, 1, 0]))


@njit(cache=True)
def multisqrt(y):
    '''Valid real solutions to y=x*x.'''
    if y < 0:
        return [float(x) for x in range(0)]
    elif y==0:
        return [0.]
    else:
        sqrty = np.sqrt(y)
        return [-sqrty, sqrty]


@njit(cache=True)
def _maybe_swap(G, swapXY):
    if swapXY:
        Q = G[np.array([1,0,2]),:][:,np.array([1,0,2])]
        return Q
    else:
        return G


@njit(cache=True)
def factor_degenerate(G, zero=0):
    '''Linear factors of a degenerate quadratic polynomial.'''
    if G[0,0] == 0 == G[1,1]:
        return [[G[0,1], 0, G[1,2]],
                [0, G[0,1], G[0,2] - G[1,2]]]

    swapXY = abs(G[0,0]) > abs(G[1,1])
    Q = _maybe_swap(G, swapXY)
    Q /= Q[1,1]
    q22 = cofactor(Q.real,(2,2))

    if -q22 <= zero:
        lines = [[Q[0,1], Q[1,1], Q[1,2]+s] for s in multisqrt(-cofactor(Q.real,(0,0)))]
    else:
        x0,y0 = [cofactor(Q.real,(i,2)) / q22 for i in [0,1]]
        lines = [[m, Q[1,1], -Q[1,1]*y0 - m*x0] for m in [Q[0,1] + s for s in multisqrt(-q22)]]

    if swapXY:
        lines = [[L[1], L[0], L[2]] for L in lines]
    return lines


@njit(cache=True)
def intersections_ellipse_line(ellipse=None, line=None, zero=1e-6):
    '''Points of intersection between ellipse and line.'''
    _,V = np.linalg.eig(np.cross(line,ellipse).T)
    return [ v.real / v[2].real for v in V.T if
             abs(np.dot(np.array(line).real, v.real)) < 10**-12 and
             abs(np.dot(v.real,ellipse.real).dot(v.real)) < 10**-12]



@njit(cache=True)
def intersections_ellipses(A, B, zero=1e-6):
    '''Points of intersection between two ellipses.'''
    A = A.astype(np.complex128)
    B = B.astype(np.complex128)
    if abs(np.linalg.det(B)) > abs(np.linalg.det(A)): A,B = B,A
    eigval = [e.real for e in np.linalg.eigvals(np.linalg.inv(A).dot(B)) if np.abs(e.imag) <= zero][0]
    lines = factor_degenerate(B - eigval*A)
    points = [point for L in lines for point in intersections_ellipse_line(A,L)]
    return points


nSS_spec = {"D2": float32,
            "b": float64[:],
            "mu": float64[:],
            "mb2": float32,
            "pb": float32,
            "pmu": float32,
            "c": float32,
            "s": float32,
            "m": float32,
            "m_alt": float32,
            "x0": float32,
            "y0": float32,
            "x1": float32,
            "y1": float32,
            "Z": float32,
            "Wm2": float64}
    

@jitclass(nSS_spec)
class nuSolutionSet(object):
    '''Collection of definitions for neutrino analytic solution in t->b,mu,nu.'''
    
    def __init__(self, b, mu, # Lorentz Vectors
                 Wm2=80.385**2, Tm2=172.5**2 #GeV**2
                 ):
        self.mu = mu
        self.b = b
        pb=np.sqrt(b[0]**2+b[1]**2+b[2]**2)
        pmu=np.sqrt(mu[0]**2+mu[1]**2+mu[2]**2)
        self.mb2 = b[3] ** 2 - pb ** 2
        self.D2 = 0.5 * (Tm2 - Wm2 - self.mb2)
        self.pb =pb
        self.pmu=pmu
        self.c = (b[0]*mu[0]+b[1]*mu[1]+b[2]*mu[2])/(pb*pmu)
        if self.c>1:
            print("cos exceeds allowed values, set to 1-calculated val:",self.c)
            self.c=1
        self.s = math.sqrt(1-self.c**2)
        self.x0 = -0.5 * Wm2 / pmu
        self.y0 = - ( self.x0*self.c + self.D2 / pb ) / self.s

        self.m = (np.abs(self.b[3])/pb - self.c) / self.s
        self.m_alt = (-np.abs(self.b[3])/pb - self.c) / self.s
        self.y1 = -self.x0 / self.m
        self.x1 = self.x0 + (self.y1 - self.y0) / self.m
        
        self.Z = math.sqrt(max(0, self.y1 * (self.y1 - 2*self.y0) - self.x0**2 - Wm2))
        self.Wm2 = Wm2

    @property
    def K(self):
        '''Extended rotation from F' to F coordinates.'''
        return np.array([[self.c, -self.s, 0, 0],
                         [self.s,  self.c, 0, 0],
                         [     0,       0, 1., 0],
                         [     0,       0, 0, 1.]])

    @property
    def A_mu(self):
        '''F coordinates constraint on W momentum: paraboloid.'''
        return np.array([[0,       0, 0,                self.x0],
                         [0,       1., 0,                      0],
                         [0,       0, 1.,                      0],
                         [self.x0, 0, 0, self.Wm2 - self.x0**2]])

    @property
    def A_b(self):
        '''F coordinates constraint on W momentum: ellipsoid.'''
        K, D2, Wm2, e2 = self.K, self.D2, self.Wm2, self.b[3]**2
        return K.dot(np.array([[ self.mb2,   0,  0,  -D2*self.pb],
                               [           0,   e2, 0,               0],
                               [           0,   0, e2,               0],
                               [-D2*self.pb, 0, 0, e2 * Wm2 - D2**2]
                               ]
                              ) ).dot(K.T)

    @property
    def R_T(self):
        '''Rotation from F coordinates to laboratory coordinates.'''
        mu = self.mu
        mutheta = np.arctan2(np.sqrt(mu[0] ** 2 + mu[1] ** 2), mu[2])
        muphi = np.arctan2(mu[1], mu[0])
        R_z = R(2, -muphi)
        R_y = R(1,  0.5*math.pi - mutheta)
        R_x = [R(0,-math.atan2(z,y)) for x,y,z in (R_y.dot( R_z.dot( self.b[0:3]) ),)][0]
        return R_z.T.dot(R_y.T.dot(R_x.T))

    @property
    def E_tilde(self):
        '''Transformation of t=[c,s,1] to nu momentum: F coordinates.'''
        Z, m, x1, y1, p = self.Z, self.m, self.x1, self.y1, self.pmu
        return np.array([[ Z/m,  0,  x1 - p ],
                         [ Z,    0,      y1 ],
                         [ 0,    Z,       0 ]])
    @property
    def E(self):
        '''Transformation of t=[c,s,1] to nu momentum: lab coordinates.'''
        return self.R_T.dot(self.E_tilde)

    @property
    def E_perp(self):
        '''Transformation of t=[c,s,1] to nu transverse momentum: lab coordinates.'''
        return np.vstack((self.E[:2], np.array([[0,0,1.]])))
    
    @property
    def N(self):
        '''Solution ellipse of nu transverse momentum: lab coordinates.'''
        E = self.E_perp
        return np.linalg.inv(E.T).dot(UnitCircle()).dot(np.linalg.inv(E))


@njit(cache=True)
def nus(ts, es):
    ret_nus = []
    for e,t in zip(es,ts):
        ret_nus.append(e.dot(np.array([math.cos(t),math.sin(t),1])))
    return ret_nus


@njit(cache=True)
def test_func(es, met, params):
    ret_xy = -met[:2]
    for nu in nus(params, es):
        ret_xy += nu[:2]
    return ret_xy


#@njit(float64[:][:](float64[:][:], float64[:][:]), cache=True)
def _typed_K(E, E_perp):
    return np.dot(E, np.linalg.inv(E_perp))


#@njit(float64[:][:](float64[:][:], float64[:][:]), cache=True)
def _typed_K_(E, E_perp):
    return E.dot(np.linalg.inv(E_perp))


def _run_kinreco(l, al, b, bbar, METx, METy, Wm2=80.385**2, Tm2=172.5**2):
    nu = np.empty((3, 0), dtype=np.float64)
    nubar = np.empty((3, 0), dtype=np.float64)
    fails = 0
    nsols=np.zeros(len(b), dtype=np.int64)
    for eventi in range(len(l)):
        bs = (b[eventi], bbar[eventi])
        mus = (al[eventi], l[eventi])
        metX, metY = METx[eventi], METy[eventi]
        solutionSets = [nuSolutionSet(B, M, Wm2, Tm2) for B,M in zip(bs,mus)]

        V0 = np.outer( [metX, metY, 0 ], [0,0,1])
        S = V0 - UnitCircle()

        hasZs = solutionSets[0].Z!=0 and solutionSets[1].Z!=0
        if hasZs:    
            N,N_ = [ss.N for ss in solutionSets]
            n_ = S.T.dot(N_).dot(S)
        
            v = intersections_ellipses(N, n_)
            v_ = [S.dot(sol) for sol in v]

#            if not v and leastsq:
#                es = [ss.E_perp for ss in solutionSets]
#                met = np.array([metX,metY,1])
#                ts,_ = leastsq(partial(test_func, es, met), [0,0], ftol=5e-5, epsfcn=0.01 )
#                v,v_ = [[i] for i in nus(ts, es)]

            ss = solutionSets[0]
            print(v)
            K = _typed_K(ss.E, ss.E_perp)
            ss = solutionSets[1]
            K_ = _typed_K_(ss.E, ss.E_perp)
#            solutions = awkward.fromiter(([K.dot(s) for s in v], [K_.dot(s_) for s_ in v_]))
            nusols = np.array([K.dot(s) for s in v]).T
            nubarsols = np.array([K_.dot(s_) for s_ in v_]).T
            nu = np.concatenate((nu, nusols), axis=1)
            nubar = np.concatenate((nubar, nubarsols), axis=1)
            nsols[eventi]=len(nusols[0])
        else:
            fails += 1
    print(fails, " / ", len(nsols))
    return nu, nubar, nsols

def _makeJCA(counts, lv):
    return Jca.candidatesfromcounts(counts, px=lv.x, py=lv.y, pz=lv.z, mass=lv.mass)

def _make_2D_array(lv):
    return np.stack((lv.x, lv.y, lv.z, lv.E), axis=-1)
    

def betchart(lep, antilep, b, antib, MET, MW=80.385, Mt=172.5):
    METx=MET.pt*np.cos(MET.phi)
    METy=MET.pt*np.sin(MET.phi)
    v, vb, nsols = _run_kinreco(_make_2D_array(lep),
                                _make_2D_array(antilep),
                                _make_2D_array(b),
                                _make_2D_array(antib),
                                METx,
                                METy)
    v = Jca.candidatesfromcounts(nsols, px=v[0], py=v[1], pz=v[2], mass=0)
    print(v)
    av = Jca.candidatesfromcounts(nsols, px=vb[0], py=vb[1], pz=vb[2], mass=0)
    has_sol = nsols > 0
    l = _makeJCA(np.ones(len(nsols)), lep)
    al = _makeJCA(np.ones(len(nsols)), antilep)
    b = _makeJCA(np.ones(len(nsols)), b)
    bbar = _makeJCA(np.ones(len(nsols)), antib)
    wp = v.cross(al)
    wm = av.cross(l)
    t = wp.cross(b)
    at = wm.cross(bbar)
    min_mtt = (t + at).p4.mass.argmin()
    t = t[min_mtt]
    at = at[min_mtt]
    return t, at

if __name__ == '__main__':
    lep=vlv([26.923591], [16.170616], [-162.3227], [165.33320])
    antilep=vlv([-34.58441], [-13.27824], [-32.51431], [49.290821])
    b=vlv([99.415420], [-78.89404], [-161.6102], [205.54469])
    antib=vlv([-49.87086], [91.930526], [-347.3868], [362.82086])
#    for x in [lep, antilep, b, antib]:
#        x = _makeJCA([1], x) 
    nu=vlv([34.521587], [-51.23474], [-6.555319], [70.848953])
    antinu=vlv([11.179965], [-3.844941], [7.0419898], [13.760989])
    MET = nu + antinu
    v, v_ = betchart(lep, antilep, b, antib, MET)
    print(v.p4, v_.p4)
    
    import time
    n = 10000
    lep = uproot_methods.TLorentzVectorArray.from_cartesian(
        np.full(n, 26.923591), np.full(n, 16.170616), np.full(n, -162.3227),
        np.full(n, 165.33320))
    alep = uproot_methods.TLorentzVectorArray.from_cartesian(
        np.full(n, -34.58441), np.full(n, -13.27824), np.full(n, -32.51431),
        np.full(n, 49.290821))
    b = uproot_methods.TLorentzVectorArray.from_cartesian(
        np.full(n, 99.415420), np.full(n, -78.89404), np.full(n, -161.6102),
        np.full(n, 205.54469))
    ab = uproot_methods.TLorentzVectorArray.from_cartesian(
        np.full(n, -49.87086), np.full(n, 91.930526), np.full(n, -347.3868),
        np.full(n, 362.82086))
    met = uproot_methods.TLorentzVectorArray.from_cartesian(
        np.full(n, 45.701552), np.full(n, -55.079681), np.full(n, 0.),
        np.full(n, 71.5709656))

    t = time.time()
    betchart(lep, alep, b, ab, met)
    print("Took {} s for {} events".format(time.time() - t, n))
