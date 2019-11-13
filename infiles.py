import glob

ttbardilep=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/*/*.root")
ttbarsemilep=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/*/*/*.root")
DY10to50=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/*/*/*.root")
DY50toinf=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/*/*/*.root")
SingleTop_tch=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/NANOAODSIM/*/*/*.root")
SingleAntitop_tch=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/NANOAODSIM/*/*/*.root")
ST_sch=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-madgraph-pythia8/NANOAODSIM/*/*/*.root")
tW=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/*/*/*.root")
tbarW=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/*/*/*.root")
TTWtoLNu=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/NANOAODSIM/*/*/*.root")
TTWtoQQ=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/NANOAODSIM/*/*/*.root")
TTZtoQQ=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/*/*/*.root")
TTZto2L2Nu=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/*/*/*.root")
WtoLNu=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/*/*/*.root")
WW=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/WW_TuneCP5_13TeV-pythia8/NANOAODSIM/*/*/*.root")
WZ=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/WZ_TuneCP5_13TeV-pythia8/NANOAODSIM/*/*/*.root")
ZZ=glob.glob("/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/ZZ_TuneCP5_13TeV-pythia8/NANOAODSIM/*/*/*.root")

fileset = {"DY10to50": DY10to50,
       "DY50toinf": DY50toinf,
       "SingleTop_tch": SingleTop_tch,
       "SingleAntitop_tch": SingleAntitop_tch,
       "ST_sch": ST_sch,
       "tW": tW,
       "tbarW": tbarW,
       "ttbarsemilep": ttbarsemilep,
       "ttbardilep": ttbardilep,
       "TTWtoLNu": TTWtoLNu,
       "TTWtoQQ": TTWtoQQ,
       "TTZtoQQ": TTZtoQQ,
       "TTZto2L2Nu": TTZto2L2Nu,
       "WtoLNu": WtoLNu,
       "WW": WW,
       "WZ": WZ,
       "ZZ": ZZ}

smallfileset = {"ttbardilep":["/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/250000/14933F79-95FB-354D-A917-E19B5C005037.root"],
                "ttbarsemilep":["/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/120000/F753B5BF-A400-1045-88DA-F0536F4E29A2.root"],
                "WZ": ["/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/WZ_TuneCP5_13TeV-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/100000/19D01271-7CE1-5546-B156-B2F7C6485B9D.root"],
                "ZZ": ["/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/ZZ_TuneCP5_13TeV-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/30000/7C3ABEB1-8038-F74A-B614-977DDD498D63.root"]}

labels={"DY10to50": "Drell Yan",
        "DY50toinf": "Drell Yan",
        "SingleTop_tch": "Single top",
        "SingleAntitop_tch": "Single top",
        "ST_sch": "Single top",
        "tW": "Single top",
        "tbarW": "Single top",
        "ttbardilep": "$\\mathrm{t}\\bar{\\mathrm{t}}$ dilep",
        "ttbarsemilep": "$\\mathrm{t}\\bar{\\mathrm{t}}$ semilep",
        "TTWtoLNu": "$\\mathrm{t}\\bar{\\mathrm{t}}\\mathrm{V}$",
        "TTWtoQQ": "$\\mathrm{t}\\bar{\\mathrm{t}}\\mathrm{V}$",
        "TTZto2L2Nu": "$\\mathrm{t}\\bar{\\mathrm{t}}\\mathrm{V}$",
        "TTZtoQQ": "$\\mathrm{t}\\bar{\\mathrm{t}}\\mathrm{V}$",
        "WtoLNu": "$\\mathrm{W}+\\mathrm{jets}$",
        "WW": "$\\mathrm{VV}$",
        "WZ": "$\\mathrm{VV}$",
        "ZZ": "$\\mathrm{VV}$"}

xsecs={"DY10to50": 18610000.0,
       "DY50toinf": 6077220.0,
       "SingleTop_tch": 136020.0,
       "SingleAntitop_tch": 80950.0,
       "ST_sch": 6350.0,
       "tW": 35850.0,
       "tbarW": 35850.0,
       "ttbarsemilep": 365345.2,
       "ttbardilep": 88287.7,
       "TTWtoLNu": 214.9,
       "TTWtoQQ": 431.6,
       "TTZtoQQ": 510.4,
       "TTZto2L2Nu": 243.2,
       "WtoLNu": 61526700,
       "WW": 75800,
       "WZ": 27600,
       "ZZ": 12140}

HLTriggers={"HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "HLT_Mu27_Ele37_CaloIdL_MW",
            "HLT_Mu37_Ele27_CaloIdL_MW"}
