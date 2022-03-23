import os
import pepper
import hist as hi
import uproot


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

        self.axisname = datahist.axes[0].name
        hist_config = {
            "bins": [
                {
                    "name": self.axisname,
                    "label": ("True mean number interactions per bunch "
                              "crossing")
                }
            ],
            "fill": {
                self.axisname: [
                    "Pileup",
                    "nTrueInt"
                ]
            }
        }
        if isinstance(datahist.axes[0], hi.axis.Regular):
            ax = datahist.axes[0]
            hist_config["bins"][0].update({
                "n_or_arr": len(ax),
                "lo": ax.value(0),
                "hi": ax.value(len(ax))
            })
        else:
            hist_config["bins"][0]["n_or_arr"] = datahist.axes[0].edges
        config["hists"] = {"pileup": pepper.HistDefinition(hist_config)}
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
    def _save_hists(hist, datahist, datahistup, datahistdown, filename,
                    sum_datasets):
        datasets = ["all_datasets"] if sum_datasets else hist.axes["dataset"]

        with uproot.recreate(filename) as f:
            for dataset in datasets:
                axpos = sum if sum_datasets else dataset
                denom = hist[{"dataset": axpos}].values().copy()
                # Set bins that are zero in MC to 0 in data to get norm right
                is_nonzero = denom != 0
                # Avoid division by zero warning
                denom[~is_nonzero] = 1
                denom /= denom.sum()
                for datahist_i, suffix in [
                        (datahist, ""), (datahistup, "_up"),
                        (datahistdown, "_down")]:
                    datahist_i = datahist_i * is_nonzero
                    norm = datahist_i.sum().value
                    ratio = datahist_i / norm / denom

                    f[dataset + suffix] = ratio

    def save_output(self, output, dest):
        datahist, datahistup, datahistdown = self.load_input_hists()

        mchist = None
        for dataset, hists in output["hists"].items():
            if mchist is None:
                mchist = hists[("Before cuts", "pileup")].copy()
            else:
                mchist += hists[("Before cuts", "pileup")]
        # Set underflow and 0 pileup bin to 0, which might be != 0 only for
        # buggy reasons in MC
        axidx = [ax.name for ax in mchist.axes].index(self.axisname)
        slic = (slice(None),) * axidx + (slice(None, 2),)
        mchist.view(flow=True)[slic].fill(0)

        self._save_hists(
            mchist, datahist, datahistup, datahistdown,
            os.path.join(dest, "pileup.root"), True)
        self._save_hists(
            mchist, datahist, datahistup, datahistdown,
            os.path.join(dest, "pileup_perdataset.root"), False)


if __name__ == "__main__":
    from pepper import runproc
    runproc.run_processor(
        Processor, "Create histograms needed for pileup reweighting",
        mconly=True)
