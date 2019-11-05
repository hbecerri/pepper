from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from coffea.analysis_objects import JaggedTLorentzVectorArray
from coffea.util import awkward, numpy
from coffea.util import numpy as np
from awkward import JaggedArray
import coffea.processor as processor
from coffea.processor.parsl.parsl_executor import parsl_executor
import uproot
import matplotlib
import matplotlib.pyplot as plt
from AdUtils import concatenate, LVwhere, Pairswhere
from uproot_methods.classes.TLorentzVector import PtEtaPhiMassLorentzVector as LV
from KinRecoSonnenschein import KinReco
from betchartkinreco import BetchartKinReco, BetchartAllBcandidates
from parsl import load, python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import CondorProvider
from parsl.addresses import address_by_hostname
from parsl.channels import LocalChannel
import glob

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
      pt=df['Electron_pt'],
      eta=df['Electron_eta'],
      phi=df['Electron_phi'],
      mass=df['Electron_mass'],
      pdgId=df['Electron_pdgId'],
      dxy=df['Electron_dxy'],
      dz=df['Electron_dz'],
      EnergyErr=df['Electron_energyErr'],
      
      deltaEtaSC=df['Electron_deltaEtaSC'],
      cutBased=df['Electron_cutBased'],
      ID_MVA_Iso_80=df['Electron_mvaFall17V2Iso_WP80'],
      ID_MVA_Iso_90=df['Electron_mvaFall17V2Iso_WP90'],
      ID_MVA_noIso_80=df['Electron_mvaFall17V2noIso_WP80'],
      ID_MVA_noIso_90=df['Electron_mvaFall17V2noIso_WP90'],
      
      mediumId=numpy.empty_like(df['Electron_dz'], dtype=bool), #initialise variables which are actually for muons to allow concatenating
      tightId=numpy.empty_like(df['Electron_dz'], dtype=bool),
      looseId=numpy.empty_like(df['Electron_dz'], dtype=bool),
      pfIsoId=numpy.empty_like(df['Electron_genPartFlav']),
      highPtId=numpy.empty_like(df['Electron_genPartFlav']),
      )
    muons = JaggedCandidateArray.candidatesfromcounts(
      df['nMuon'],
      pt=df['Muon_pt'],
      eta=df['Muon_eta'],
      phi=df['Muon_phi'],
      mass=df['Muon_mass'],
      pdgId=df['Muon_pdgId'],
      dxy=df['Muon_dxy'],
      dz=df['Muon_dz'],
      EnergyErr=df['Muon_ptErr'], #note this is actually a pt err, but has been named like this for consistency with electrons
      
      deltaEtaSC=numpy.empty_like(df['Muon_dz']), #electron variables (to allow concatenating)
      cutBased=numpy.empty_like(df['Muon_pdgId']),
      ID_MVA_Iso_80=numpy.empty_like(df['Muon_dz'], dtype=bool),
      ID_MVA_Iso_90=numpy.empty_like(df['Muon_dz'], dtype=bool),
      ID_MVA_noIso_80=numpy.empty_like(df['Muon_dz'], dtype=bool),
      ID_MVA_noIso_90=numpy.empty_like(df['Muon_dz'], dtype=bool),
      
      mediumId=df['Muon_mediumId'],
      tightId=df['Muon_tightId'],
      looseId=df['Muon_looseId'],
      pfIsoId=df['Muon_pfIsoId'],
      highPtId=df['Muon_highPtId'],
      )
    jets = JaggedCandidateArray.candidatesfromcounts(
      df['nJet'],
      pt=df['Jet_pt'],
      eta=df['Jet_eta'],
      phi=df['Jet_phi'],
      mass=df['Jet_mass'],
      btag=df['Jet_btagDeepB'],
      )
    genpart = JaggedCandidateArray.candidatesfromcounts(
      df['nGenPart'],
      pt=df['GenPart_pt'],
      eta=df['GenPart_eta'],
      phi=df['GenPart_phi'],
      mass=np.where((np.abs(df['GenPart_pdgId'])==5)&(df['GenPart_mass']==0),4.8, df['GenPart_mass']), #mass is only stored for masses greater than 10GeV- this is a fudge to set the b mass to the right value if not stored
      pdgId=df['GenPart_pdgId'],
      motherIdx=df['GenPart_genPartIdxMother'],
      status=df['GenPart_status'],
      statusFlags=df['GenPart_statusFlags'],
      )
    MET= awkward.Table(pt=df['MET_pt'], phi=df['MET_phi'])
    genMET= awkward.Table(pt=df['GenMET_pt'], phi=df['GenMET_phi'])
    
    leptons = concatenate([electrons, muons]) #(Could instead use unionarray, but this causes problems with e.g. acessing the lorentz vectors-see https://github.com/scikit-hep/awkward-array/issues/149)
    if len(leptons)==0: return output #Occasionally parsl can create empty chunks, which cause problems: skip these
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
    jetsnob=jets[jets.btag<=0.4184]
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
    
    b0, b1=Pairswhere(btags.counts>1, btags.distincts(), btags.cross(jetsnob))
    
    bs=concatenate([b0, b1])
    bbars=concatenate([b1, b0])
    
    GenHistFile=uproot.open("/nfs/dust/cms/user/stafford/coffea/desy-ttbarbsm-coffea/GenHists.root")
    HistMlb=GenHistFile["Mlbbar"]
    alb=bs.cross(leadantilep)
    lbbar=bbars.cross(leadantilep)
    PMalb=JaggedArray.fromcounts(bs.counts, HistMlb.allvalues[np.searchsorted(HistMlb.alledges, alb.mass.content)-1])
    PMlbbar=JaggedArray.fromcounts(bs.counts, HistMlb.allvalues[np.searchsorted(HistMlb.alledges, lbbar.mass.content)-1])
    bestbpairMlb=(PMalb*PMlbbar).argmax()
    b=bs[bestbpairMlb]
    bbar=bbars[bestbpairMlb]
    
    #print("Events pre reco:", len(leadlep))
    neutrino, antineutrino=BetchartKinReco(leadlep['p4'], leadantilep['p4'], b['p4'], bbar['p4'], MET)
    Reco=neutrino.counts>0
    output['cutflow']['Reconstruction']+=Reco.sum()
    #print("Reco events", Reco.sum())
    rneutrino=neutrino[Reco]
    rantineutrino=antineutrino[Reco]
    rleadlep=leadlep[Reco]
    rleadantilep=leadantilep[Reco]
    rb=b[Reco]
    rbbar=bbar[Reco]
    Wminus=rantineutrino.cross(rleadlep)
    Wplus=rneutrino.cross(rleadantilep)
    top=Wplus.cross(rb)
    antitop=Wminus.cross(rbbar)
    ttbar=top['p4']+antitop['p4']
    best=ttbar.mass.argmin()
    Mttbar=ttbar[best].mass.content
    MWplus=Wplus[best].mass.content
    Mtop=top[best].mass.content
    
    output['Mttbar'].fill(dataset="Reco", Mttbar=Mttbar)
    output['Mt'].fill(dataset="Reco", Mt=Mtop)
    output['MWplus'].fill(dataset="Reco", MW=MWplus)
    
    genlep=genpart[((genpart.pdgId==11)|(genpart.pdgId==13))&(genpart[genpart.motherIdx].pdgId==-24)]#&(genpart[genpart[genpart.motherIdx].motherIdx].pdgId==-6)]&(genpart.statusFlags>>12&1==1)]
    genantilep=genpart[((genpart.pdgId==-11)|(genpart.pdgId==-13))&(genpart[genpart.motherIdx].pdgId==24)]#&(genpart[genpart[genpart.motherIdx].motherIdx].pdgId==6)]#&(genpart.statusFlags>>12&1==1)]
    toomanyleps=genlep[genlep.counts>1]
    genb=genpart[(genpart.pdgId==5)&(genpart[genpart.motherIdx].pdgId==6)&(genpart.statusFlags>>12&1==1)&(genpart.statusFlags>>7&1==1)]
    print("no. gen bs:", genb.counts.sum())
    genantib=genpart[(genpart.pdgId==-5)&(genpart[genpart.motherIdx].pdgId==-6)&(genpart.statusFlags>>12&1==1)&(genpart.statusFlags>>7&1==1)]
    twogenleps=(genlep.counts==1)&(genantilep.counts==1)&(genb.counts==1)&(genantib.counts==1)
    genlep=genlep[twogenleps]
    genantilep=genantilep[twogenleps]
    genb=genb[twogenleps]
    genantib=genantib[twogenleps]
    gennu=genpart[((genpart.pdgId==12)|(genpart.pdgId==14))&(genpart[genpart.motherIdx].pdgId==24)]
    genantinu=genpart[((genpart.pdgId==-12)|(genpart.pdgId==-14))&(genpart[genpart.motherIdx].pdgId==-24)]
    gennu=gennu[twogenleps]
    genantinu=genantinu[twogenleps]
    reconu, recoantinu=BetchartKinReco(genlep['p4'], genantilep['p4'], genb['p4'], genantib['p4'], genMET[twogenleps])
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
    print("genreco eff:", hassoln.sum()/len(genb))
    
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
    
    t=genpart[(genpart.pdgId==6)&(genpart.statusFlags>>12&1==1)]
    tbar=genpart[(genpart.pdgId==-6)&(genpart.statusFlags>>12&1==1)]
    print("nt", (t.counts).sum(), (tbar.counts).sum())
    ttbar=t['p4']+tbar['p4']
    output['Mttbar'].fill(dataset="Top", Mttbar=ttbar.mass.content)
    output['Mt'].fill(dataset="Top", Mt=t.mass.content)
    
    #genl=genpart[((genpart.pdgId==11)|(genpart.pdgId==13))]
    #output['lep_parents']=
    
    return output

  def postprocess(self, accumulator):
    return accumulator


#source /nfs/dust/cms/user/stafford/coffea/cmssetup.sh
wrk_init='''
export PATH=/afs/desy.de/user/s/stafford/.local/bin:$PATH
export PYTHONPATH=/afs/desy.de/user/s/stafford/.local/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=/nfs/dust/cms/user/stafford/coffea/desy-ttbarbsm-coffea:$PYTHONPATH
'''

config = Config(
    executors=[HighThroughputExecutor(label="HTCondor",
                                      address=address_by_hostname(),
				      prefetch_capacity=0,
				      cores_per_worker=1,
				      max_workers=10,
				      provider=CondorProvider(channel=LocalChannel(),
				                              init_blocks=100,
							      min_blocks=5,
							      max_blocks=1000,
							      nodes_per_block=1,
							      parallelism=1,
							      scheduler_options='''Requirements = OpSysAndVer == "CentOS7"''', 
							      worker_init=wrk_init))],
    lazy_errors=False
)
dfk=load(config)

ttbar=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/250000/*.root")
ttbar.append(glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/60000/*.root"))
fileset = {'TTbar': ttbar}

'''output = processor.run_uproot_job(fileset,
                                 treename='Events',
                                 processor_instance=ttbarProcessor(),
                                 executor=processor.iterative_executor,
				 executor_args={'workers': 6, 'flatten': True},
                                 chunksize = 100000)
'''
output = processor.run_parsl_job(fileset,
                                 treename='Events',
                                 processor_instance=ttbarProcessor(),
                                 executor=parsl_executor,
                                 chunksize = 100000)

cutvals=np.array(list(output['cutflow'].values()))
cuteffs=100*cutvals[1:]/cutvals[:-1]

print("in:", cutvals[0])
print("out:", cutvals[-1])

plt.bar(np.arange(len(cuteffs)), cuteffs)
ax = plt.gca()

ax.set_xticks(np.arange(len(cuteffs)))
ax.set_xticklabels(np.array(list(output['cutflow'].keys()))[1:])
ax.set_ylabel('Efficiency')

plt.show(block=True)

plot=hist.plot1d(output['Mttbar'], overlay='dataset', density=1)#, error_opts={"yerr":None})

plt.show(block=True)

plot=hist.plot1d(output['MWplus'], overlay='dataset', density=1)

plt.show(block=True)

plot=hist.plot1d(output['Mt'], overlay='dataset', density=1)
plt.show(block=True)
