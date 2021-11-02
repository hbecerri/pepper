import os
import pepper
import coffea
import uproot
import numpy as np


class Processor(pepper.Processor):
    def __init__(self, config, eventdir):
        self.data_pu_hist = config["data_pu_hist"]
        self.data_pu_hist_up = config["data_pu_hist_up"]
        self.data_pu_hist_down = config["data_pu_hist_down"]

        datahist, datahistup, datahistdown = self.load_input_hists()
        if len(datahist.axes) != 1:
            raise pepper.config.ConfigError(
                "data_pu_hist has invalid number of axes. Only one axis is "
                "allowed.")
        if datahist.axes != datahistup.axes:
            raise pepper.config.ConfigError(
                "data_pu_hist_up does not have the same axes as "
                "data_pu_hist.")
        if datahist.axes != datahistdown.axes:
            raise pepper.config.ConfigError(
                "data_pu_hist_down does not have the same axes as "
                "data_pu_hist.")

        axisname = datahist.axes[0].name
        config["hists"] = {
            "pileup": pepper.HistDefinition({
                "bins": [
                    {
                        "name": axisname,
                        "label": ("True mean number interactions per bunch "
                                  "crossing"),
                        "n_or_arr": datahist.axes[0].edges
                    }
                ],
                "fill": {
                    axisname: [
                        "Pileup",
                        "nTrueInt"
                    ]
                }
            })
        }
        if "hists_to_do" in config:
            del config["hists_to_do"]
        config["compute_systematics"] = False
        # Treat all datasets as normal datasets, instead of using them as
        # systematic
        config["dataset_for_systematics"] = {}

        super().__init__(config, eventdir)

    def preprocess(self, datasets):
        # Only run over MC
        processed = {}
        for key, value in datasets.items():
            if key in self.config["mc_datasets"]:
                processed[key] = value
        return processed

    def setup_selection(self, data, dsname, is_mc, filler):
        # Ignore generator weights, because pileup is independent
        return pepper.Selector(data, on_update=filler.get_callbacks())

    def process_selection(self, selector, dsname, is_mc, filler):
        pass

    def load_input_hists(self):
        with uproot.open(self.data_pu_hist) as f:
            datahist = f["pileup"].to_hist()
        with uproot.open(self.data_pu_hist_up) as f:
            datahistup = f["pileup"].to_hist()
        with uproot.open(self.data_pu_hist_down) as f:
            datahistdown = f["pileup"].to_hist()
        return datahist, datahistup, datahistdown

    @staticmethod
    def _save_hists(hist, datahist, datahistup, datahistdown, filename):
        hist = pepper.misc.coffeahist2hist(hist)

        with uproot.recreate(filename) as f:
            for dataset in hist.axes["dataset"]:
                hist_int = hist[dataset, :]
                hist_int /= hist_int.sum().value
                for datahist_i, suffix in [
                        (datahist, ""), (datahistup, "_up"),
                        (datahistdown, "_down")]:
                    ratio = datahist_i / hist_int.values()
                    # Set infinities from zero height bins in hist_int to 0
                    ratio[:] = np.nan_to_num(
                        np.stack([ratio.values(), ratio.variances()], axis=-1),
                        posinf=0)
                    f[dataset + suffix] = ratio

    def save_output(self, output, dest):
        datahist, datahistup, datahistdown = self.load_input_hists()

        # Normalize data histograms
        datahist /= datahist.sum().value
        datahistup /= datahistup.sum().value
        datahistdown /= datahistdown.sum().value

        mchist = output["hists"][("Before cuts", "pileup")]
        # Set underflow and 0 pileup bin to 0, which might be != 0 only for
        # buggy reasons in MC
        for idn in mchist.identifiers("dataset"):
            mchist._sumw[(idn,)][:2] = 0

        new_ds_axis = coffea.hist.Cat("dataset", "Dataset")
        mchist_allds = mchist.group(
            mchist.axis("dataset"), new_ds_axis,
            {"all_datasets": [i.name for i in mchist.identifiers("dataset")]})
        self._save_hists(
            mchist_allds, datahist, datahistup, datahistdown,
            os.path.join(dest, "pileup.root"))
        self._save_hists(
            mchist, datahist, datahistup, datahistdown,
            os.path.join(dest, "pileup_perdataset.root"))


if __name__ == "__main__":
    from pepper import runproc
    runproc.run_processor(
        Processor, "Create histograms needed for pileup reweighting",
        mconly=True)
