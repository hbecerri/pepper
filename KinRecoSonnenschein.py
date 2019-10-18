from coffea.util import numpy as np
import uproot_methods
from coffea.analysis_objects import JaggedCandidateArray
from coffea.util import awkward


def KinReco(lep, antilep, b, antib, MET, verbosity=0): #note:inputs should be Lorentz vectors rather than jagged arrays (as it's easier to get just the Lorentz vector for the second b quark)
  lE=lep.E.flatten()
  lx=lep.x.flatten()
  ly=lep.y.flatten()
  lz=lep.z.flatten()
  alE=antilep.E.flatten()
  alx=antilep.x.flatten()
  aly=antilep.y.flatten()
  alz=antilep.z.flatten()
  bE=b.E.flatten()
  bx=b.x.flatten()
  by=b.y.flatten()
  bz=b.z.flatten()
  abE=antib.E.flatten()
  abx=antib.x.flatten()
  aby=antib.y.flatten()
  abz=antib.z.flatten()
  METx=MET.pt*np.cos(MET.phi)
  METy=MET.pt*np.sin(MET.phi)
  
  mt = 172.5
  mtb = 172.5
  mWp = 80.3
  mWm = 80.3
  mb=b.mass.flatten()
  mbb=antib.mass.flatten()
  ml=lep.mass.flatten()
  mal=antilep.mass.flatten()
  
  #Reduce all constraints to a single equation by Sonnenschein's method: https://arxiv.org/pdf/hep-ph/0603011.pdf
  a1=(bE+alE)*(mWp**2-mal**2)-alE*(mt**2-mb**2-mal**2)+2*bE*alE**2-2*alE*(bx*alx+by*aly+bz*alz)
  a2=2*(bE*alx-alE*bx)
  a3=2*(bE*aly-alE*by)
  a4=2*(bE*alz-alE*bz)
  
  b1=(abE+lE)*(mWm**2-ml**2)-lE*(mtb**2-mb**2-ml**2)+2*abE*lE**2-2*lE*(abx*lx+aby*ly+abz*lz)
  b2=2*(abE*lx-lE*abx)
  b3=2*(abE*ly-lE*aby)
  b4=2*(abE*lz-lE*abz)
  
  c00=-4*(alE**2-aly**2)-4*(alE**2-alz**2)*(a3/a4)**2-8*aly*alz*a3/a4
  c10=-8*(alE**2-alz**2)*a2*a3/(a4**2)+8*alx*aly-8*alx*alz*a3/a4-8*aly*alz*a2/a4
  c11=4*(mWp**2-mal**2)*(aly-alz*a3/a4)-8*(alE**2-alz**2)*a1*a3/(a4**2)-8*aly*alz*a1/a4
  c20=-4*(alE**2-alx**2)-4*(alE**2-alz**2)*(a2/a4)**2-8*alx*alz*a2/a4
  c21=4*(mWp**2-mal**2)*(alx-alz*a2/a4)-8*(alE**2-alz**2)*a1*a2/(a4**2)-8*alx*alz*a1/a4
  c22=(mWp**2-mal**2)**2-4*(alE**2-alz**2)*(a1/a4)**2-4*mWp**2*alz*a1/a4
  
  d00=-4*(lE**2-ly**2)-4*(lE**2-lz**2)*(b3/b4)**2-8*ly*lz*b3/b4
  d10=-8*(lE**2-lz**2)*b2*b3/(b4**2)+8*lx*ly-8*lx*lz*b3/b4-8*ly*lz*b2/b4
  d11p=4*(mWm**2-ml**2)*(ly-lz*b3/b4)-8*(lE**2-lz**2)*b1*b3/(b4**2)-8*ly*lz*b1/b4
  d20=-4*(lE**2-lx**2)-4*(lE**2-lz**2)*(b2/b4)**2-8*lx*lz*b2/b4
  d21p=4*(mWm**2-ml**2)*(lx-lz*b2/b4)-8*(lE**2-lz**2)*b1*b2/(b4**2)-8*lx*lz*b1/b4
  d22p=(mWm**2-ml**2)**2-4*(lE**2-lz**2)*(b1/b4)**2-4*mWm**2*lz*b1/b4
  
  d11=-d11p-2*METy*d00-METx*d10
  d21=-d21p-2*METx*d20-METy*d10
  d22=d22p+METx**2*d20+METy**2*d00+METx*METy*d10+METx*d21p+METy*d11p
  
  h0=c00**2*d20**2+c10*d20*(c10*d00-c00*d10)+c20*d10*(c00*d10-c10*d00)+c20*d00*(c20*d00-2*c00*d20)
  h1=c00*d21*(2*c00*d20-c10*d10)-c00*d20*(c11*d10+c10*d11)+c00*d10*(2*c20*d11+c21*d10)-2*c00*d00*(c21*d20+c20*d21)+c10*d00*(2*c11*d20+c10*d21)+c20*d00*(2*c21*d00-c10*d11)-d00*d10*(c11*c20+c10*c21) #note the sign of c20*d0*(...) is different to the appendix of Sonnenschein's paper, and instead follows the implementation on github https://github.com/gerbaudo/ttbar-kinsol-comp, which gives the right solution
  h2=c00**2*(2*d22*d20+d21**2)-c00*d21*(c11*d10+c10*d11)+c11*d20*(c11*d00-c00*d11)+c00*d10*(c22*d10-c10*d22)+c00*d11*(2*c21*d10+c20*d11)+(2*c22*c20+c21**2)*d00**2-2*c00*d00*(c22*d20+c21*d21+c20*d22)+c10*d00*(2*c11*d21+c10*d22)-d00*d10*(c11*c21+c10*c22)-d00*d11*(c11*c20+c10*c21)
  h3=c00*d21*(2*c00*d22-c11*d11)+c00*d11*(2*c22*d10+c21*d11)+c22*d00*(2*c21*d00-c11*d10)-c00*d22*(c11*d10+c10*d11)-2*c00*d00*(c22*d21+c21*d22)-d00*d11*(c11*c21+c10*c22)+c11*d00*(c11*d21+2*c10*d22)
  h4=c00**2*d22**2+c11*d22*(c11*d00-c00*d11)+c00*c22*(d11**2-2*d00*d22)+c22*d00*(c22*d00-c11*d11)
  h=np.array([h0, h1, h2, h3, h4])
  H=np.swapaxes(h, 0, 1)
  #H=H.reshape(-1, 5)
  
  if(verbosity==1):
    print("a1=", a1[0], "a2=", a2[0], "a3=", a3[0], "a4=", a4[0])
    print("b1=", b1[0], "b2=", b2[0], "b3=", b3[0], "b4=", b4[0])
    print("c00=", a4[0]**2*c00[0], "c10=", a4[0]**2*c10[0], "c11=", a4[0]**2*c11[0], "c20=", a4[0]*c20[0], "c21=", a4[0]**2*c21[0], "c22=", a4[0]**2*c22[0])
    print("d00=", b4[0]**2*d00[0], "d10=", b4[0]**2*d10[0], "d11=", b4[0]**2*d11[0], "d20=", b4[0]*d20[0], "d21=", b4[0]**2*d21[0], "d22=", b4[0]**2*d22[0])
    print("h0=", a4[0]**4*b4[0]**4*h0[0], "h1=", a4[0]**4*b4[0]**4*h1[0], "h2=", a4[0]**4*b4[0]**4*h2[0], "h3=", a4[0]**4*b4[0]**4*h3[0], "h4=", a4[0]**4*b4[0]**4*h4[0])
  
  
  neutrinopx=[]
  nsols=np.zeros(len(lep), dtype=int)
  for eventi in range(len(lep)):
    if(np.isfinite(H[eventi]).sum()==5):
      roots=np.roots(H[eventi])
      for root in roots:
        if np.abs(root.imag)<10**-6: #(an arbritary accuracy for imaginary part to account for possible numerical errors-consider revising)
          neutrinopx.append(root)
          nsols[eventi]+=1
    else:
      print(H[eventi])
      print(lx[eventi], ly[eventi], lz[eventi])
      print(alx[eventi], aly[eventi], alz[eventi])
      print(bx[eventi], by[eventi], bz[eventi])
      print(abx[eventi], aby[eventi], abz[eventi])
      print(a4[eventi])
    if (a4[eventi]==0): print("Panic!")
  neutrinopx=np.array(neutrinopx, dtype=float)
  vpx=awkward.JaggedArray.fromcounts(nsols, neutrinopx)
  c0=c00
  c1=c10*vpx+c11
  c2=c20*vpx**2+c21*vpx+c22
  d0=d00
  d1=d10*vpx+d11
  d2=d20*vpx**2+d21*vpx+d22
  vpy=(c0*d2-c2*d0)/(c1*d0-c0*d1)
  
  vpz=(-a1-a2*vpx-a3*vpy)/a4
  
  vbarpx=METx-vpx
  vbarpy=METy-vpy
  
  vbarpz=(-b1-b2*vbarpx-b3*vbarpy)/b4
  
  neutrino=JaggedCandidateArray.candidatesfromcounts(nsols, px=vpx.content, py=vpy.content, pz=vpz.content, mass=0)
  antineutrino=JaggedCandidateArray.candidatesfromcounts(nsols, px=vbarpx.content, py=vbarpy.content, pz=vbarpz.content, mass=0)
  return neutrino, antineutrino

if __name__ == '__main__':
  # test case:
  lep=JaggedCandidateArray.candidatesfromcounts([1], energy=np.array([165.33320]), px=np.array([26.923591]), py=np.array([16.170616]), pz=np.array([-162.3227]))
  antilep=JaggedCandidateArray.candidatesfromcounts([1], energy=np.array([49.290821]), px=np.array([-34.58441]), py=np.array([-13.27824]), pz=np.array([-32.51431]))
  b=JaggedCandidateArray.candidatesfromcounts([1], energy=np.array([205.54469]), px=np.array([99.415420]), py=np.array([-78.89404]), pz=np.array([-161.6102]))
  antib=JaggedCandidateArray.candidatesfromcounts([1], energy=np.array([362.82086]), px=np.array([-49.87086]), py=np.array([91.930526]), pz=np.array([-347.3868]))
  nu=JaggedCandidateArray.candidatesfromcounts([1], energy=np.array([70.848953]), px=np.array([34.521587]), py=np.array([-51.23474]), pz=np.array([-6.555319]))
  antinu=JaggedCandidateArray.candidatesfromcounts([1], energy=np.array([13.760989]), px=np.array([11.179965]), py=np.array([-3.844941]), pz=np.array([7.0419898]))
  MET= awkward.Table(pt=(nu['p4']+antinu['p4']).pt[0], phi=(nu['p4']+antinu['p4']).phi[0])
  neutrino, antineutrino=KinReco(lep['p4'], antilep['p4'], b['p4'], antib['p4'], MET)
  print(neutrino['p4'].x)
