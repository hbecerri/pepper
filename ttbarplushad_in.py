import glob

ttbardilep=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/*/*.root")
ttbarsemilep=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/*/*/*.root")
ttbarhad=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/*/*/*.root")

fileset = {
       "ttbarsemilep": ttbarsemilep,
       "ttbardilep": ttbardilep,
       "ttbarhad": ttbarhad}

smallfileset = {"ttbardilep":["/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/250000/14933F79-95FB-354D-A917-E19B5C005037.root"],
                "ttbarsemilep":["/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/120000/F753B5BF-A400-1045-88DA-F0536F4E29A2.root"],
                "ttbarhad": ["/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/120000/FE4C29C4-5698-3D45-AD06-A0DBA455347F.root"]}

labels={
        "ttbardilep": "$\\mathrm{t}\\bar{\\mathrm{t}}$ dilep",
        "ttbarsemilep": "$\\mathrm{t}\\bar{\\mathrm{t}}$ semilep",
        "ttbarhad": "$\\mathrm{t}\\bar{\\mathrm{t}}$ had"}

xsecs={
       "ttbarsemilep": 365345.2,
       "ttbardilep": 88287.7,
       "ttbarhad": 377960.7}
