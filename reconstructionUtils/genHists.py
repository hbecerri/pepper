from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from coffea.analysis_objects import JaggedTLorentzVectorArray
from coffea.util import awkward
from coffea.util import numpy
from coffea.util import numpy as np
import uproot
import coffea.processor as processor
from coffea.processor.parsl.parsl_executor import parsl_executor
import matplotlib
import matplotlib.pyplot as plt
from parsl import load, python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import CondorProvider
from parsl.addresses import address_by_hostname
from parsl.channels import LocalChannel
import glob

class ttbarProcessor(processor.ProcessorABC):
    def __init__(self):
        Mlb_axis = hist.Bin("Mlb", "Mlb [GeV]", 100, 0, 200)

        
        self._accumulator = processor.dict_accumulator({
            'Mlbarb': hist.Hist("Counts", Mlb_axis),
            'Mlbbar': hist.Hist("Counts", Mlb_axis)
        })
    
    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, df):
        output = self.accumulator.identity()
        
        dataset = df['dataset']
        if df['nGenPart'].sum()==0:    return output
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
        #genMET= awkward.Table(pt=df['GenMET_pt'], phi=df['GenMET_phi'])
        
        genlep=genpart[((genpart.pdgId==11)|(genpart.pdgId==13))
                       &(genpart[genpart.motherIdx].pdgId==-24)
                       &(genpart.statusFlags>>12&1==1)]
        genantilep=genpart[((genpart.pdgId==-11)|(genpart.pdgId==-13))
                           &(genpart[genpart.motherIdx].pdgId==24)
                           &(genpart.statusFlags>>12&1==1)]
        genb=genpart[(genpart.pdgId==5)&(genpart[genpart.motherIdx].pdgId==6)
                     &(genpart.statusFlags>>12&1==1)&(genpart.statusFlags>>7&1==1)]
        genantib=genpart[(genpart.pdgId==-5)&(genpart[genpart.motherIdx].pdgId==-6)
                         &(genpart.statusFlags>>12&1==1)&(genpart.statusFlags>>7&1==1)]
        twogenleps=(genlep.counts==1)&(genantilep.counts==1)
                   &(genb.counts==1)&(genantib.counts==1)
        genlep=genlep[twogenleps]
        genantilep=genantilep[twogenleps]
        genb=genb[twogenleps]
        genantib=genantib[twogenleps]
        Mlbarb=(genantilep['p4']+genb['p4']).mass.content
        Mlbbar=(genlep['p4']+genantib['p4']).mass.content
        
        output['Mlbarb'].fill(Mlb=Mlbarb)
        output['Mlbbar'].fill(Mlb=Mlbbar)
        return output
        
    def postprocess(self, accumulator):
        return accumulator

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

ttbar=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/250000/*.root")+glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/60000/*.root")

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

outfile = uproot.recreate('/nfs/dust/cms/user/stafford/coffea/desy-ttbarbsm-coffea/data/GenHists.root')
outfile['Mlbbar'] = hist.export1d(output['Mlbbar'])
outfile['Mlbarb'] = hist.export1d(output['Mlbarb'])
outfile['Mlb'] = hist.export1d(output['Mlbbar'].add(output['Mlbarb']))
outfile.close()
