from coffea import hist
import numpy as np

channels=["ee", "emu", "mumu", "None"]

def fill_MET(data, dsname, is_mc, weight):
    dataset_axis = hist.Cat("dataset", "")
    MET_axis = hist.Bin("MET", "MET [GeV]", 100, 0, 400)
    if "Channel" in data:
        channel_axis = hist.Cat("channel", "")
        MET_hist = hist.Hist("Counts", dataset_axis, channel_axis, MET_axis)
        if is_mc:
            for ch in channels:
                vals = np.where(data["Channel"] == ch)
                MET_hist.fill(dataset=dsname,
                              channel=ch,
                              MET=data["MET_pt"][vals],
                              weight=weight[vals])
        else:
            for ch in channels:
                vals = np.where(data["Channel"] == ch)
                MET_hist.fill(dataset=dsname,
                              channel=ch,
                              MET=data["MET_pt"][vals])
    else:
        MET_hist = hist.Hist("Counts", dataset_axis, MET_axis)
        if is_mc:
            MET_hist.fill(dataset=dsname,
                          MET=data["MET_pt"],
                          weight=weight)
        else:
            MET_hist.fill(dataset=dsname, MET=data["MET_pt"])
    return MET_hist

def fill_puppi_MET(data, dsname, is_mc, weight):
    dataset_axis = hist.Cat("dataset", "")
    MET_axis = hist.Bin("MET", "MET [GeV]", 100, 0, 400)
    if "Channel" in data:
        channel_axis = hist.Cat("channel", "")
        MET_hist = hist.Hist("Counts", dataset_axis, channel_axis, MET_axis)
        if is_mc:
            for ch in channels:
                vals = np.where(data["Channel"] == ch)
                MET_hist.fill(dataset=dsname,
                              channel=ch,
                              MET=data["PuppiMET_pt"][vals],
                              weight=weight[vals])
        else:
            for ch in channels:
                vals = np.where(data["Channel"] == ch)
                MET_hist.fill(dataset=dsname,
                              channel=ch,
                              MET=data["PuppiMET_pt"][vals])
    else:
        MET_hist = hist.Hist("Counts", dataset_axis, MET_axis)
        if is_mc:
            MET_hist.fill(dataset=dsname,
                          MET=data["PuppiMET_pt"],
                          weight=weight)
        else:
            MET_hist.fill(dataset=dsname, MET=data["PuppiMET_pt"])
    return MET_hist

def fill_MET_sig(data, dsname, is_mc, weight):
    dataset_axis = hist.Cat("dataset", "")
    MET_axis = hist.Bin("MET_sig", "MET significance", 100, 0, 600)
    if "Channel" in data:
        channel_axis = hist.Cat("channel", "")
        MET_hist = hist.Hist("Counts", dataset_axis, channel_axis, MET_axis)
        if is_mc:
            for ch in channels:
                vals = np.where(data["Channel"] == ch)
                MET_hist.fill(dataset=dsname,
                              channel=ch,
                              MET_sig=data["MET_significance"][vals],
                              weight=weight[vals])
        else:
            for ch in channels:
                vals = np.where(data["Channel"] == ch)
                MET_hist.fill(dataset=dsname,
                              channel=ch,
                              MET_sig=data["MET_significance"][vals])
    else:
        MET_hist = hist.Hist("Counts", dataset_axis, MET_axis)
        if is_mc:
            MET_hist.fill(dataset=dsname,
                          MET_sig=data["MET_significance"],
                          weight=weight)
        else:
            MET_hist.fill(dataset=dsname, MET_sig=data["MET_significance"])
    return MET_hist


def fill_Mll(data, dsname, is_mc, weight):
    dataset_axis = hist.Cat("dataset", "")
    Mll_axis = hist.Bin("Mll", "Mll [GeV]", 100, 0, 400)
    Mll_hist = hist.Hist("Counts", dataset_axis, Mll_axis)
    if "Mll" in data:
        if is_mc:
            Mll_hist.fill(dataset=dsname, Mll=data["Mll"],
                          weight=data["genWeight"])
        else:
            Mll_hist.fill(dataset=dsname, Mll=data["Mll"])
    return Mll_hist

hist_dict = {"MET": fill_MET,
             "Puppi_MET": fill_puppi_MET,
             "MET_sig": fill_MET_sig}