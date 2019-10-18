from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from coffea.analysis_objects import JaggedTLorentzVectorArray
from coffea.util import awkward
from coffea.util import numpy
from coffea.util import numpy as np
import coffea.processor as processor
import matplotlib
import matplotlib.pyplot as plt
from AdUtils import concatenate, LVwhere
from uproot_methods.classes.TLorentzVector import PtEtaPhiMassLorentzVector as LV
from KinRecoSonnenschein import KinReco

matplotlib.interactive(True)

class ttbarProcessor(processor.ProcessorABC):
  def __init__(self):
    dataset_axis = hist.Cat("dataset", "")
    ttbarmass_axis = hist.Bin("Mttbar", "Mttbar [GeV]", 100, 0, 1000)
    Wmass_axis = hist.Bin("MW", "MW [GeV]", 100, 0, 200)
    tmass_axis = hist.Bin("Mt", "Mt [GeV]", 100, 0, 300)
    neutrinop = hist.Bin("neup", "neutrino px [GeV]", 100, -400, 400)
    
    self._accumulator = processor.dict_accumulator({
      'Mttbar': hist.Hist("Counts", dataset_axis, ttbarmass_axis),
      'MWplus': hist.Hist("Counts", dataset_axis, Wmass_axis),
      'Mt': hist.Hist("Counts", dataset_axis, tmass_axis),
      'neutrinopz' : hist.Hist("", dataset_axis, neutrinop),
      #'Reconeutrinopz' : hist.Hist("", neutrinop),
      'cutflow': processor.defaultdict_accumulator(int)
    })
  
  @property
  def accumulator(self):
    return self._accumulator
  
  def process(self, df):
    output = self.accumulator.identity()
    
    dataset = df['dataset']
    electrons = JaggedCandidateArray.candidatesfromcounts(
      df['nElectron'],
      pt=df['Electron_pt'].content,
      eta=df['Electron_eta'].content,
      phi=df['Electron_phi'].content,
      mass=df['Electron_mass'].content,
      pdgId=df['Electron_pdgId'].content,
      dxy=df['Electron_dxy'].content,
      dz=df['Electron_dz'].content,
      EnergyErr=df['Electron_energyErr'].content,
      
      deltaEtaSC=df['Electron_deltaEtaSC'].content,
      cutBased=df['Electron_cutBased'].content,
      ID_MVA_Iso_80=df['Electron_mvaFall17V2Iso_WP80'].content,
      ID_MVA_Iso_90=df['Electron_mvaFall17V2Iso_WP90'].content,
      ID_MVA_noIso_80=df['Electron_mvaFall17V2noIso_WP80'].content,
      ID_MVA_noIso_90=df['Electron_mvaFall17V2noIso_WP90'].content,
      
      mediumId=numpy.empty_like(df['Electron_dz'].content, dtype=bool), #initialise variables which are actually for muons to allow concatenating
      tightId=numpy.empty_like(df['Electron_dz'].content, dtype=bool),
      looseId=numpy.empty_like(df['Electron_dz'].content, dtype=bool),
      pfIsoId=numpy.empty_like(df['Electron_genPartFlav'].content),
      highPtId=numpy.empty_like(df['Electron_genPartFlav'].content),
      )
    muons = JaggedCandidateArray.candidatesfromcounts(
      df['nMuon'],
      pt=df['Muon_pt'].content,
      eta=df['Muon_eta'].content,
      phi=df['Muon_phi'].content,
      mass=df['Muon_mass'].content,
      pdgId=df['Muon_pdgId'].content,
      dxy=df['Muon_dxy'].content,
      dz=df['Muon_dz'].content,
      EnergyErr=df['Muon_ptErr'].content, #note this is actually a pt err, but has been named like this for consistency with electrons
      
      deltaEtaSC=numpy.empty_like(df['Muon_dz'].content), #electron variables (to allow concatenating)
      cutBased=numpy.empty_like(df['Muon_pdgId'].content),
      ID_MVA_Iso_80=numpy.empty_like(df['Muon_dz'].content, dtype=bool),
      ID_MVA_Iso_90=numpy.empty_like(df['Muon_dz'].content, dtype=bool),
      ID_MVA_noIso_80=numpy.empty_like(df['Muon_dz'].content, dtype=bool),
      ID_MVA_noIso_90=numpy.empty_like(df['Muon_dz'].content, dtype=bool),
      
      mediumId=df['Muon_mediumId'].content,
      tightId=df['Muon_tightId'].content,
      looseId=df['Muon_looseId'].content,
      pfIsoId=df['Muon_pfIsoId'].content,
      highPtId=df['Muon_highPtId'].content,
      )
    jets = JaggedCandidateArray.candidatesfromcounts(
      df['nJet'],
      pt=df['Jet_pt'].content,
      eta=df['Jet_eta'].content,
      phi=df['Jet_phi'].content,
      mass=df['Jet_mass'].content,
      btag=df['Jet_btagDeepB'].content,
      )
    genpart = JaggedCandidateArray.candidatesfromcounts(
      df['nGenPart'],
      pt=df['GenPart_pt'].content,
      eta=df['GenPart_eta'].content,
      phi=df['GenPart_phi'].content,
      mass=np.where((np.abs(df['GenPart_pdgId'].content)==5)&(df['GenPart_mass'].content==0),4.8, df['GenPart_mass'].content), #mass is only stored for masses greater than 10GeV- this is a fudge to set the b mass to the right value if not stored
      pdgId=df['GenPart_pdgId'].content,
      motherIdx=df['GenPart_genPartIdxMother'].content,
      status=df['GenPart_status'].content,
      statusFlags=df['GenPart_statusFlags'].content,
      )
    MET= awkward.Table(pt=df['MET_pt'], phi=df['MET_phi'])
    genMET= awkward.Table(pt=df['GenMET_pt'], phi=df['GenMET_phi'])
    
    leptons = concatenate([electrons, muons]) #(Could instead use unionarray, but this causes problems with e.g. acessing the lorentz vectors-see https://github.com/scikit-hep/awkward-array/issues/149)
    leps=leptons[leptons.pdgId>0]
    antileps=leptons[leptons.pdgId<0]
    
    output['cutflow']['Events preselection'] += leptons.size

    leadlep=leps[leps.pt.argmax()]
    leadantilep=antileps[antileps.pt.argmax()]
    
    #Lepton Event cuts
    lepPair=((leps.counts>0)&(antileps.counts>0))
    output['cutflow']['Lep pair'] += lepPair.sum()
    
    exactly2leps=((leps.counts==1)&(antileps.counts==1))
    output['cutflow']['Exactly 2 leps'] += exactly2leps.sum()
    
    leadleppt=numpy.zeros(len(leps), dtype=bool)
    leadleppt[leps.counts>0]=(leadlep.pt>25).flatten()
    leadantileppt=numpy.zeros(len(antileps), dtype=bool)
    leadantileppt[antileps.counts>0]=(leadantilep.pt>25).flatten()
    leadinglepcut=numpy.logical_or(leadleppt,leadantileppt)
    
    lepcuts=lepPair&exactly2leps&leadinglepcut
    output['cutflow']['Leading lepton pt cut']+=lepcuts.sum()
    leadlep=leadlep[lepcuts]
    leadantilep=leadantilep[lepcuts]
    jets=jets[lepcuts]
    MET=MET[lepcuts]
    
    dilepP4=leadlep.cross(leadantilep)
    mindilepM=(dilepP4.mass>20)
    output['cutflow']['Mll>20GeV']+=mindilepM.sum().sum()
    
    invZcut=(leadlep.pdgId!=-leadantilep.pdgId)|((dilepP4.mass<75)|(dilepP4.mass>106))
    dilepcuts=(mindilepM&invZcut).flatten()
    output['cutflow']['Inverse Z cut']+=dilepcuts.sum()
    
    leadlep=leadlep[dilepcuts]
    leadantilep=leadantilep[dilepcuts]
    jets=jets[dilepcuts]
    MET=MET[dilepcuts]
    
    #Jet Event cuts
    jets=jets[jets.pt>20]
    twogoodjets=jets[jets.pt>30].counts>2
    output['cutflow']['Two good jets']+=twogoodjets.sum()
    leadlep=leadlep[twogoodjets]
    leadantilep=leadantilep[twogoodjets]
    jets=jets[twogoodjets]
    MET=MET[twogoodjets]
    
    #MET cuts    
    MET_cut=MET.pt>40
    output['cutflow']['MET pt>40GeV']+=MET_cut.sum()
    leadlep=leadlep[MET_cut]
    leadantilep=leadantilep[MET_cut]
    jets=jets[MET_cut]
    MET=MET[MET_cut]
    
    #Require at least 1 b tag
    btags=jets[jets.btag>0.4184]#Deep CSV medium working point
    jetsnob=jets[jets.btag<0.4184]
    Btagcut=btags.counts>0
    output['cutflow']['At least 1 btag']+=Btagcut.sum()
    leadlep=leadlep[Btagcut]
    leadantilep=leadantilep[Btagcut]
    jetsnob=jetsnob[Btagcut]
    jets=jets[Btagcut]
    MET=MET[Btagcut]
    btags=btags[Btagcut]
    
    #Reject events with MET aligned with jets
    METnotalignedwithjets=(np.abs((jets.phi - MET.phi + np.pi) % (2*np.pi) - np.pi)).max()>1.5
    output['cutflow']['MET jet align']+=METnotalignedwithjets.sum()
    leadlep=leadlep[METnotalignedwithjets]
    leadantilep=leadantilep[METnotalignedwithjets]
    jetsnob=jetsnob[METnotalignedwithjets]
    MET=MET[METnotalignedwithjets]
    btags=btags[METnotalignedwithjets]
    
    bjet1=btags[:,0]
    twobtags=btags[btags.counts>1]
    jetsnob=jetsnob[btags.counts<2]
    bjet2=LVwhere(btags.counts>1, twobtags[:,1]['p4'], jetsnob[:,0]['p4'])
    
    neutrino, antineutrino=KinReco(leadlep['p4'], leadantilep['p4'], bjet1['p4'], bjet2['p4'], MET)
    Reco=neutrino.counts>1
    output['cutflow']['Reconstruction']+=Reco.sum()
    neutrino=neutrino[Reco]
    antineutrino=antineutrino[Reco]
    leadlep=leadlep[Reco]
    leadantilep=leadantilep[Reco]
    bjet1=bjet1[Reco]
    bjet2=bjet2[Reco]
    Wminus=antineutrino.cross(leadlep)
    Wplus=neutrino.cross(leadantilep)
    b=JaggedCandidateArray.candidatesfromcounts(np.ones(len(bjet1)), pt=bjet1['p4'].pt,eta=bjet1['p4'].eta,phi=bjet1['p4'].phi,mass=bjet1['p4'].mass)
    top=Wplus.cross(b)
    antitop=Wminus.cross(bjet2)
    ttbar=top['p4']+antitop['p4']
    best=ttbar.mass.argmin()
    Mttbar=ttbar[best].mass.content
    MWplus=Wplus[best].mass.content
    Mtop=top[best].mass.content
    
    output['Mttbar'].fill(dataset="Reco", Mttbar=Mttbar)
    output['Mt'].fill(dataset="Reco", Mt=Mtop)
    output['MWplus'].fill(dataset="Reco", MW=MWplus)
    
    genlep=genpart[((genpart.pdgId==11)|(genpart.pdgId==13))&(genpart[genpart.motherIdx].pdgId==-24)&(genpart[genpart[genpart.motherIdx].motherIdx].pdgId==-6)&(genpart.statusFlags>>12&1==1)]
    genantilep=genpart[((genpart.pdgId==-11)|(genpart.pdgId==-13))&(genpart[genpart.motherIdx].pdgId==24)&(genpart[genpart[genpart.motherIdx].motherIdx].pdgId==6)&(genpart.statusFlags>>12&1==1)]
    genb=genpart[(genpart.pdgId==5)&(genpart[genpart.motherIdx].pdgId==6)&(genpart.statusFlags>>12&1==1)]
    genantib=genpart[(genpart.pdgId==-5)&(genpart[genpart.motherIdx].pdgId==-6)&(genpart.statusFlags>>12&1==1)&(genpart.statusFlags>>7&1==1)]
    twogenleps=(genlep.counts>0)&(genantilep.counts>0)
    genlep=genlep[twogenleps]
    genantilep=genantilep[twogenleps]
    genb=genb[twogenleps]
    genantib=genantib[twogenleps]
    gennu=genpart[((genpart.pdgId==12)|(genpart.pdgId==14))&(genpart[genpart.motherIdx].pdgId==24)&(genpart[genpart[genpart.motherIdx].motherIdx].pdgId==6)]
    genantinu=genpart[((genpart.pdgId==-12)|(genpart.pdgId==-14))&(genpart[genpart.motherIdx].pdgId==-24)&(genpart[genpart[genpart.motherIdx].motherIdx].pdgId==-6)]
    gennu=gennu[twogenleps]
    genantinu=genantinu[twogenleps]
    reconu, recoantinu=KinReco(genlep['p4'], genantilep['p4'], genb['p4'], genantib['p4'], genMET[twogenleps])
    Wplus=reconu.cross(genantilep)
    Wminus=recoantinu.cross(genlep)
    top=Wplus.cross(genb)
    antitop=Wminus.cross(genantib)
    ttbar=top['p4']+antitop['p4']
    print((ttbar.counts>0).sum())
    best=ttbar.mass.argmin()
    Mttbar=ttbar[best].mass.content
    MWplus=Wplus[best].mass.content
    Mtop=top[best].mass.content
    reconu=reconu[best]
    hassoln=(ttbar.counts>0)
    gennuforsols=gennu[hassoln, 0]
    
    output['Mttbar'].fill(dataset="Gen Reco", Mttbar=Mttbar)
    output['Mt'].fill(dataset="Gen Reco", Mt=Mtop)
    output['MWplus'].fill(dataset="Gen Reco", MW=MWplus)
    Wplus=gennu.cross(genantilep)
    Wminus=genantinu.cross(genlep)
    top=Wplus.cross(genb)
    antitop=Wminus.cross(genantib)
    ttbar=top.cross(antitop)
    print((ttbar.counts>0).sum())
    best=ttbar.mass.argmin()
    Mttbar=ttbar[best].mass.content
    MWplus=Wplus[best].mass.content
    Mtop=top[best].mass.content
    gennu=gennu[best]
    
    output['Mttbar'].fill(dataset="Gen", Mttbar=Mttbar)
    output['Mt'].fill(dataset="Gen", Mt=Mtop)
    output['MWplus'].fill(dataset="Gen", MW=MWplus)
    '''print(len(gennu.content))
    print(len(reconu.content))
    output['neutrinopz'].fill(dataset="gen", neup=(gennu.content)['p4'].x)
    output['neutrinopz'].fill(dataset="reco", neup=(reconu.content)['p4'].x)'''
    
    return output

  def postprocess(self, accumulator):
    return accumulator

fileset = {'Jets': ["/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/250000/5239A9C7-9C9D-2B4B-B337-374252B493F0.root"]}

output = processor.run_uproot_job(fileset,
                                 treename='Events',
                                 processor_instance=ttbarProcessor(),
                                 executor=processor.iterative_executor,
                                 executor_args={'workers':4},
                                 chunksize = 500000)

cutvals=np.array(list(output['cutflow'].values()))
cuteffs=100*cutvals[1:]/cutvals[:-1]

plt.bar(np.arange(len(cuteffs)), cuteffs)
ax = plt.gca()

ax.set_xticks(np.arange(len(cuteffs)))
ax.set_xticklabels(np.array(list(output['cutflow'].keys()))[1:])
ax.set_ylabel('Efficiency')

plt.show(block=True)

plot=hist.plot1d(output['Mttbar'], overlay='dataset', density=1)

plt.show(block=True)

plot=hist.plot1d(output['MWplus'], overlay='dataset', density=1)

plt.show(block=True)

plot=hist.plot1d(output['Mt'], overlay='dataset', density=1)
plt.show(block=True)

