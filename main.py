from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from coffea.analysis_objects import JaggedTLorentzVectorArray
from coffea.util import awkward, numpy
from coffea.util import numpy as np
from awkward import JaggedArray
import coffea.processor as processor
import coffea
import uproot
import matplotlib
import matplotlib.pyplot as plt
from uproot_methods.classes.TLorentzVector import PtEtaPhiMassLorentzVector as LV
from parsl import load, python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import CondorProvider
from parsl.addresses import address_by_hostname
from parsl.channels import LocalChannel

from functools import partial
from collections import defaultdict
#import htcondor
import logging

from AdUtils import concatenate, LVwhere, Pairswhere
from reconstructionUtils.betchartkinreco import BetchartKinReco, BetchartAllBcandidates
from reconstructionUtils.KinRecoSonnenschein import KinReco
from infiles import *
#from ttbarplushad_in import *

logging.basicConfig(filename='log.txt', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

matplotlib.interactive(True)


class ttbarProcessor(processor.ProcessorABC):
    Defaultdictint=partial(processor.defaultdict_accumulator, int)
    
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
            'cutflow': processor.defaultdict_accumulator(self.Defaultdictint)
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
            eta=df['Electron_eta']+df['Electron_deltaEtaSC'],
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
        if df['nJet'].sum()==0:  return output #Occasionally parsl can create empty chunks, which cause problems: skip these
        jets = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt=df['Jet_pt'],
            eta=df['Jet_eta'],
            phi=df['Jet_phi'],
            mass=df['Jet_mass'],
            btag=df['Jet_btagDeepB'],
            )
        '''genpart = JaggedCandidateArray.candidatesfromcounts(
            df['nGenPart'],
            pt=df['GenPart_pt'],
            eta=df['GenPart_eta'],
            phi=df['GenPart_phi'],
            mass=np.where((np.abs(df['GenPart_pdgId'])==5)&(df['GenPart_mass']==0),4.8, df['GenPart_mass']), #mass is only stored for masses greater than 10GeV- this is a fudge to set the b mass to the right value if not stored
            pdgId=df['GenPart_pdgId'],
            motherIdx=df['GenPart_genPartIdxMother'],
            status=df['GenPart_status'],
            statusFlags=df['GenPart_statusFlags'],
            )'''
        MET= awkward.Table(pt=df['MET_pt'], phi=df['MET_phi'])
        genMET= awkward.Table(pt=df['GenMET_pt'], phi=df['GenMET_phi'])
        weights=df['Generator_weight']  #or genWeight? Not sure what the difference between them is...
        triggers=awkward.fromiter({trigger:df[trigger] for trigger in HLTriggers})
        triggered=np.any(np.array([df[trigger] for trigger in HLTriggers]), axis=0)
        
        output['cutflow'][dataset]['Events preselection'] += weights.sum()
        
        weights=weights[triggered]
        output['cutflow'][dataset]['Trigger']+=weights.sum()
        electrons=electrons[triggered]
        muons=muons[triggered]
        jets=jets[triggered]
        MET=MET[triggered]
        
        etacut=(np.abs(electrons.eta)<2.4)&((np.abs(electrons.eta)<1.4442)|(np.abs(electrons.eta)>1.5660))
        electrons=electrons[etacut]
        electrons=electrons[electrons.ID_MVA_Iso_90]
        
        muons=muons[np.abs(muons.eta)<2.4]
        muons=muons[(muons.mediumId)&(muons.pfIsoId>1)]
        
        leptons = concatenate([electrons, muons]) #(Could instead use unionarray, but this causes problems with e.g. acessing the lorentz vectors-see https://github.com/scikit-hep/awkward-array/issues/149)
        leps=leptons[leptons.pdgId>0]
        antileps=leptons[leptons.pdgId<0]
        
        leadlep=leps[leps.pt.argmax()]
        leadantilep=antileps[antileps.pt.argmax()]
        
        #Lepton Event cuts
        lepPair=((leps.counts>0)&(antileps.counts>0))
        output['cutflow'][dataset]['Lep pair'] += (weights[lepPair]).sum()
        
        exactly2leps=((leps.counts==1)&(antileps.counts==1))
        output['cutflow'][dataset]['Exactly 2 leps'] += (weights[exactly2leps]).sum()
        
        leadleppt=numpy.zeros(len(leps), dtype=bool)
        leadleppt[leps.counts>0]=(leadlep.pt>25).flatten()
        leadantileppt=numpy.zeros(len(antileps), dtype=bool)
        leadantileppt[antileps.counts>0]=(leadantilep.pt>25).flatten()
        leadinglepcut=numpy.logical_or(leadleppt,leadantileppt)
        
        lepcuts=lepPair&exactly2leps&leadinglepcut
        weights=weights[lepcuts]
        output['cutflow'][dataset]['Leading lepton pt cut']+=weights.sum()
        leadlep=leadlep[lepcuts]
        leadantilep=leadantilep[lepcuts]
        jets=jets[lepcuts]
        MET=MET[lepcuts]
        
        dilepP4=leadlep.cross(leadantilep)
        mindilepM=(dilepP4.mass>20)
        output['cutflow'][dataset]['Mll>20GeV']+=(weights[mindilepM.flatten()]).sum()
        
        invZcut=(leadlep.pdgId!=-leadantilep.pdgId)|((dilepP4.mass<75)|(dilepP4.mass>106))
        dilepcuts=(mindilepM&invZcut).flatten()
        weights=weights[dilepcuts]
        output['cutflow'][dataset]['Inverse Z cut']+=weights.sum()
        
        leadlep=leadlep[dilepcuts]
        leadantilep=leadantilep[dilepcuts]
        jets=jets[dilepcuts]
        MET=MET[dilepcuts]
        
        #Jet Event cuts
        jets=jets[jets.pt>20]
        twogoodjets=jets[jets.pt>30].counts>2
        weights=weights[twogoodjets]
        output['cutflow'][dataset]['Two good jets']+=weights.sum()
        leadlep=leadlep[twogoodjets]
        leadantilep=leadantilep[twogoodjets]
        jets=jets[twogoodjets]
        MET=MET[twogoodjets]
        
        #MET cuts    
        MET_cut=MET.pt>40
        weights=weights[MET_cut]
        output['cutflow'][dataset]['MET pt>40GeV']+=weights.sum()
        leadlep=leadlep[MET_cut]
        leadantilep=leadantilep[MET_cut]
        jets=jets[MET_cut]
        MET=MET[MET_cut]
        
        #Require at least 1 b tag
        btags=jets[jets.btag>0.4184]#Deep CSV medium working point
        jetsnob=jets[jets.btag<=0.4184]
        Btagcut=btags.counts>0
        weights=weights[Btagcut]
        output['cutflow'][dataset]['At least 1 btag']+=weights.sum()
        leadlep=leadlep[Btagcut]
        leadantilep=leadantilep[Btagcut]
        jetsnob=jetsnob[Btagcut]
        jets=jets[Btagcut]
        MET=MET[Btagcut]
        btags=btags[Btagcut]
        
        #Reject events with MET aligned with jets
        METnotalignedwithjets=(np.abs((jets.phi - MET.phi + np.pi) % (2*np.pi) - np.pi)).max()>1.5
        weights=weights[METnotalignedwithjets]
        output['cutflow'][dataset]['MET jet align']+=weights.sum()
        leadlep=leadlep[METnotalignedwithjets]
        leadantilep=leadantilep[METnotalignedwithjets]
        jetsnob=jetsnob[METnotalignedwithjets]
        MET=MET[METnotalignedwithjets]
        btags=btags[METnotalignedwithjets]
        
        
        try:
            b0, b1=Pairswhere(btags.counts>1, btags.distincts(), btags.cross(jetsnob))
        
            bs=concatenate([b0, b1])
            bbars=concatenate([b1, b0])
        
            GenHistFile=uproot.open("/nfs/dust/cms/user/stafford/coffea/desy-ttbarbsm-coffea/reconstructionUtils/GenHists.root")
            HistMlb=GenHistFile["Mlb"]
            alb=bs.cross(leadantilep)
            lbbar=bbars.cross(leadantilep)
            PMalb=JaggedArray.fromcounts(bs.counts, HistMlb.allvalues[np.searchsorted(HistMlb.alledges, alb.mass.content)-1])
            PMlbbar=JaggedArray.fromcounts(bs.counts, HistMlb.allvalues[np.searchsorted(HistMlb.alledges, lbbar.mass.content)-1])
            bestbpairMlb=(PMalb*PMlbbar).argmax()
            b=bs[bestbpairMlb]
            bbar=bbars[bestbpairMlb]
        
            #print("Events pre reco:", len(leadlep))
            neutrino, antineutrino=KinReco(leadlep['p4'], leadantilep['p4'], b['p4'], bbar['p4'], MET)
            Reco=neutrino.counts>0
            weights=weights[Reco]
            output['cutflow'][dataset]['Reconstruction']+=weights.sum()
            #print("Reco events", Reco.sum())
            neutrino=neutrino[Reco]
            antineutrino=antineutrino[Reco]
            leadlep=leadlep[Reco]
            leadantilep=leadantilep[Reco]
            b=b[Reco]
            bbar=bbar[Reco]
            Wminus=antineutrino.cross(leadlep)
            Wplus=neutrino.cross(leadantilep)
            top=Wplus.cross(b)
            antitop=Wminus.cross(bbar)
            ttbar=top['p4']+antitop['p4']
            best=ttbar.mass.argmin()
            Mttbar=ttbar[best].mass.content
            MWplus=Wplus[best].mass.content
            Mtop=top[best].mass.content
        
            output['Mttbar'].fill(dataset=dataset, Mttbar=Mttbar, weight=weights)
            output['Mt'].fill(dataset=dataset, Mt=Mtop, weight=weights)
            output['MWplus'].fill(dataset=dataset, MW=MWplus, weight=weights)
        except Exception as err:
            logger.error(err)
        finally:
            return output

    def postprocess(self, accumulator):
        return accumulator

#source /nfs/dust/cms/user/stafford/coffea/cmssetup.sh
wrk_init='''
export PATH=/afs/desy.de/user/s/stafford/.local/bin:$PATH
export PYTHONPATH=/afs/desy.de/user/s/stafford/.local/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=/nfs/dust/cms/user/stafford/coffea/desy-ttbarbsm-coffea:$PYTHONPATH
'''

nproc=1
condor_cfg='''
Requirements = OpSysAndVer == "CentOS7"
RequestMemory=%d
RequestCores=%d
'''%(2048*nproc, nproc)

config = Config(
        executors=[HighThroughputExecutor(label="HTCondor",
                    address=address_by_hostname(),
                    prefetch_capacity=0,
                    cores_per_worker=1,
                    max_workers=nproc,
                    provider=CondorProvider(channel=LocalChannel(),
                        init_blocks=20,
                        min_blocks=5,
                        max_blocks=1000,
                        nodes_per_block=1,
                        parallelism=1,
                        scheduler_options=condor_cfg, 
                        worker_init=wrk_init))],
        lazy_errors=False
)
dfk=load(config)

try:
    '''output = processor.run_uproot_job(smallfileset,
                                      treename='Events',
                                      processor_instance=ttbarProcessor(),
                                      executor=processor.iterative_executor,
                                      executor_args={'workers': 4, 'flatten': True},
                                      chunksize = 100000)
    '''
    output = processor.run_uproot_job(smallfileset,
                                     treename='Events',
                                     processor_instance=ttbarProcessor(),
                                     executor=coffea.processor.parsl_executor,
                                     executor_args={'flatten': True},
                                     chunksize = 100000)
except KeyboardInterrupt:
    raise
cutvalues=dict((k,np.zeros(len(output['cutflow']["ttbardilep"])))for k in set(labels.values()))
cuteffs=dict((k,np.zeros(len(output['cutflow']["ttbardilep"])-1))for k in set(labels.values()))
lumifactors=defaultdict(int)
for dataset in fileset.keys():
    cutvals=np.array(list(output['cutflow'][dataset].values()))
    
    if len(cutvals)==0:
        eff=0
        lumifactors[dataset]=0
    else:
        eff =cutvals[-1]/cutvals[0]
        lumifactors[dataset]=0.05*xsecs[dataset]/cutvals[0]
    print (dataset, "efficiency:", eff*100)
    

    if len(cutvals>0):
        cutvalues[labels[dataset]]+=cutvals
labelsset=list(set(labels.values()))
nlabels=len(labelsset)
ax = plt.gca()
for n, label in enumerate(labelsset):
    cuteffs[label]=100*cutvalues[label][1:]/cutvalues[label][:-1]
    ax.bar(np.arange(len(cuteffs[label]))+(2*n-nlabels)*0.4/nlabels, cuteffs[label], 0.8/nlabels, label=label)

ax.set_xticks(np.arange(len(cuteffs[labels["ttbardilep"]])))
ax.set_xticklabels(np.array(list((output['cutflow']["ttbardilep"]).keys()))[1:])
ax.set_ylabel('Efficiency')

handles, labs = ax.get_legend_handles_labels() #https://stackoverflow.com/questions/43348348/pyplot-legend-index-error-tuple-index-out-of-range
leghandles=[]
leglabs=[]
for i, h in enumerate(handles):
    if len(h):
        leghandles.append(h)
        leglabs.append(labs[i])
ax.legend(leghandles, leglabs)

print(output['cutflow'].keys())
output['Mttbar'].scale(lumifactors, axis='dataset')
labelmap=defaultdict(list)
for key, val in labels.items():
    cutvals=np.array(list(output['cutflow'][key].values()))
    if len(cutvals)>0 and cutvals[-1]>0:
        labelmap[val].append(key)


sortedlabels=sorted(labelsset, key=(lambda x : sum([(output['Mttbar'].integrate('Mttbar')).values()[(y,)] for y in labelmap[x]])))
for key in sortedlabels:
    labelmap[key]=labelmap.pop(key)

labels_axis = hist.Cat("labels", "", sorting='placement')
mttbar=output['Mttbar'].group('dataset', labels_axis, labelmap)

plot, ax, _ = hist.plot1d(mttbar, overlay='labels', stack=True)

plt.show(block=True)
