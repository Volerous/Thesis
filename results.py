#%%
import pandas as pd
import glob

latex_kwargs = dict(position="H",
                    escape=False,
                    float_format="%.3f",
                    na_rep="-",
                    multirow=True,
                    multicolumn=True)


def bold_max(x: pd.Series):
    x = round(x, 3)
    x_ret = x.apply(round_vals)
    if "mean" in x.name and "Loss" not in x.name:
        x_ret[x.idxmax()] = f"\\textbf{ {x[x.idxmax()]} }"
    else:
        x_ret[x.idxmin()] = f"\\textbf{ {x[x.idxmin()]} }"
    return x_ret


def round_vals(x):
    return f"{x:.3f}" if "nan" != f"{x:.3f}" else "-"


def load_glob(val):
    files = glob.glob(val)
    dfs = []
    for i in files:
        splits = i.split("-")
        vals = {
            "Dimension": splits[1],
            "Probability": splits[2],
            "Alpha": splits[3],
            "i": splits[4]
        }

        dfs.append(pd.read_csv(i).join(pd.DataFrame(pd.Series(vals)).T))
    return pd.concat(dfs).drop(columns=["epoch", "fg_f1"])


#%% Hyper Results 1
mi: pd.DataFrame = load_glob(
    "./logs/UNet3D_aug-*-[012]-evaluate.csv").set_index(
        ["Dimension", "Probability", "Alpha", "i"])
mi = mi.rename(columns=lambda x: x.capitalize()
               if "fg_" not in x else x[3:].capitalize(), )

mi.groupby(['Dimension', 'Probability', 'Alpha']).aggregate([
    'mean', 'std'
]).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/hyper-1-mean.tex",
    caption=
    "Full results from the first set of hyperparameter tuning over the values of the dimension, alpha upper bound, and application probability",
    label="tab:hyper-1-results")
mi.sort_index().applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/hyper-1-full.tex",
    caption=
    "Hyperparameter (H.P) search full results from the first set with 3D U-Net using data augmentation",
    label="tab:hyper-1-results-full")
#%% Hyper Results 2
files = glob.glob("./logs/UNet3D_hyper-*[012]-evaluate.csv")
hyper_2 = pd.concat([pd.read_csv(i) for i in files
                     ]).drop(columns=["epoch", "fg_f1"]).sort_index()
idxs = [(1, 3, 0), (1, 3, 1), (1, 3, 2), (1, 4, 0), (1, 4, 1), (1, 4, 2),
        (1, 5, 0), (1, 5, 1), (1, 5, 2), (2, 3, 0), (2, 3, 1), (2, 3, 2)]
idxs = pd.MultiIndex.from_tuples(idxs,
                                 names=["Complexity Factor", "Depth", "i"])
hyper_2.index = idxs
hyper_2 = hyper_2.rename(columns=lambda x: x.capitalize()
                         if "fg_" not in x else x[3:].capitalize(), )
# hyper_2 = hyper_2[(1,3)]
idx = pd.IndexSlice
# hyper_2.loc[idx[1,3]] = mi.loc["96", "0.333", "100", :].to_numpy()
hyper_2.loc[idx[:, :4, :]].groupby(["Complexity Factor", "Depth"]).agg([
    'mean', 'std'
]).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/hyper-2-mean.tex",
    caption=
    "Hyperparameter (H.P) search results from the second set with 3D U-Net using data augmentation",
    label="tab:hyper-2-results",
)

hyper_2.loc[idx[:, :4, :]].applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/hyper-2-full.tex",
    caption=
    "Full results from the second set of hyperparameter tuning over the values of the complexity factor and depth, the higher complexity factor and depth values ran in to out-of-memory errors",
    label="tab:hyper-2-results-full")
hyper_2.loc[idx[1, 5, :]].groupby(["Complexity Factor", "Depth"]).agg([
    'mean', 'std'
]).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/depth-results-mean.tex",
    caption=
    "Mean Metrics from training the \"3D U-Net, H.P. 1\" modified with depth $5$ 3 times",
    label="tab:depth-results",
)

hyper_2.loc[idx[1, 5, :]].applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/depth-results-full.tex",
    caption=
    "Full Metrics from training the \"3D U-Net, H.P. 1\" modified with depth $5$ 3 times",
    label="tab:depth-results-full")

#%% Test results
files = glob.glob("./logs/*-final-evaluate.csv")
files.sort()
files = [
    i.replace("-final", "") for i in files
    if "pad-normal" not in i and "noaug" not in i and "scaled" not in i
]

test_res = pd.concat(
    [pd.read_csv(i)
     for i in files]).drop(columns=["epoch", "fg_f1", "model"]).sort_index()
idxs = [[
    "3D U-Net H.P. tuning set 1", "3D U-Net, D.A.",
    "3D U-Net H.P. tuning set 2", "3D U-Net, padding", "3D U-Net, shrinking",
    "2D U-Net", "MPUNet"
],
        list(range(3))]
idxs = pd.MultiIndex.from_product(idxs, names=["Model", "i"])
test_res.index = idxs
test_res = test_res.rename(columns=lambda x: x.capitalize()
                           if "fg_" not in x else x[3:].capitalize(), )
test_res.groupby("Model").agg(['mean', 'std']).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/test-results-mean.tex",
    caption=
    "Performance results from various models, running only with validation data and optimizations.",
    label="tab:test-results")
test_res.applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/test-results-full.tex",
    caption="Full results from the models when tested on the test set",
    label="tab:test-results-full")
test_res.loc[[
    "2D U-Net", "3D U-Net, padding", "3D U-Net, shrinking", "MPUNet"
]].groupby("Model").agg(['mean', 'std']).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/test-results-first.tex",
    caption=
    "Early performance results on models created with train and validation data, predicting on testing data. The best performing model is the \"3D U-Net, shrinking\", whereas the worst performing model is \"2D U-Net\"",
    label="tab:test-results-first")
test_res.loc[["3D U-Net, shrinking", "3D U-Net, D.A."]].groupby("Model").agg([
    'mean', 'std'
]).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/test-results-da.tex",
    caption=
    "Early performance results on models with and without Data Augmentation, (D.A.), while \"3D U-Net, shrinking\" has been previously seen in \\Fref{tab:test-results-first}",
    label="tab:test-results-data-augmentation")
# %%
idxs = [[
    "3D U-Net H.P. tuning set 1", "3D U-Net, D.A.",
    "3D U-Net H.P. tuning set 2", "3D U-Net, padding", "3D U-Net, shrinking",
    "2D U-Net", "MPUNet", "MPUNet, no D.A."
],
        list(range(3))]
idxs = pd.MultiIndex.from_product(idxs, names=["Model", "i"])
files = glob.glob("./logs/*-final-evaluate.csv")
files.sort()
files = [i for i in files if "pad-normal" not in i and "scaled" not in i]
eval_res = pd.concat([pd.read_csv(i) for i in files
                      ]).drop(columns=["epoch", "fg_f1", "model"])

eval_res.index = idxs
eval_res = eval_res.rename(columns=lambda x: x.capitalize()
                           if "fg_" not in x else x[3:].capitalize(), )
eval_res_data = eval_res.copy(True)
eval_res.drop(index=["MPUNet, no D.A."], inplace=True)
eval_res.groupby(["Model"]).agg(['mean', 'std']).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/eval-results-mean.tex",
    caption=
    "Performance results from various models that where optimized on from validation data, results shown in this table is from testing with the never before seen evaluation dataset, nothing was changed after the retreival of these results",
    label="tab:eval-results")
eval_res.applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/eval-results-full.tex",
    caption=
    "Full results from the models when tested on the final evaluation set",
    label="tab:eval-results-full")
eval_res_data.loc[["MPUNet", "MPUNet, no D.A."]].fillna("-").groupby([
    "Model"
]).agg(['mean', 'std']).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/data-augmentaion-results-mean.tex",
    caption=
    "Mean metrics $\\pm1$ standard deviation from training 3 times with the MPUnet model trained without data augmentation against the results from the original MPUnet trained with data augmentation\\Fref{tab:eval-results}, on the evaluation dataset",
    label="tab:data-augmentation-results")
eval_res_data.loc[["MPUNet, no D.A."]].applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/data-augmentaion-results-full.tex",
    caption=
    "Full metrics on the evaluation dataset from each training of the MPUnet model trained without data augmentation",
    label="tab:data-augmentation-results-full")
#%%
files = glob.glob("./logs/UNet*_*scaled-[012]-final-evaluate.csv")
files += glob.glob("./logs/UNet*_*normal-[012]-final-evaluate.csv")
files.sort()
scaled_res = pd.concat([pd.read_csv(i) for i in files
                        ]).drop(columns=["epoch", "fg_f1"]).sort_index()
idxs = [["3D U-Net, padding", "2D U-Net"], list(range(3))]
idxs = pd.MultiIndex.from_product(idxs, names=["Model", "i"])
scaled_res.index = idxs
scaled_res = scaled_res.rename(columns=lambda x: x.capitalize()
                               if "fg_" not in x else x[3:].capitalize(), )
scaled_res.applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/scaled-results-full.tex",
    caption=
    "The full results from the models trained with Normalization and standardization",
    label="tab:scaled-results-full",
)
scaled_res.groupby(["Model"]).agg(['mean', 'std']).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/scaled-results-mean.tex",
    caption=
    "The results from the models trained with Normalization and Standardization",
    label="tab:scaled-results-mean",
)
#%%
from collections import Counter

genders = [
    "M", "M", "F", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "F", "F",
    "F", "M", "F", "M", "M", "F", "F", "F", "M", "M", "M", "M", "M", "F", "F",
    "F", "M", "M", "M", "M", "M", "M", "M", "F", "F", "M", "M", "F", "M", "M",
    "F", "M", "F", "M", "M", "M", "M", "M", "F", "M", "F", "M", "F", "F", "M",
    "M"
]
splits = [("Train", 26), ("Validation", 9), ("Test", 9), ("Evaluation", 17)]
curr = 0
dict_val = {}
for (name, add_val) in splits:
    counts = Counter(genders[curr:curr + add_val])
    curr += add_val
    dict_val[name] = counts
counts = Counter(genders)
dict_val["Total"] = counts
data = pd.DataFrame(dict_val).set_axis(["M", 'F']).T
data[["M\%", "F\%"]] = (data.iloc[:, :] / data.sum(axis=1)[:, None])
data = data.sort_index(axis=1)
data.to_latex(
    **latex_kwargs,
    buf="./tables/men-women-split.tex",
    caption=
    "The split-wise and total percentage of the data that is men vs women",
    label="tab:men-women-per-split")
# %%
files = glob.glob("./logs/*data*-final-evaluate.csv")
files += glob.glob("./logs/UNet3D_pad-robust-scaled-*-final-evaluate.csv")
files.sort()
files = [i for i in files if "pad-normal" not in i and "noaug" not in i]
data_res = pd.concat([pd.read_csv(i) for i in files
                      ]).drop(columns=["epoch", "fg_f1"]).sort_index()
idxs = [[
    "3D U-Net, D.A., Data", "3D U-Net, HP 1, Data", "3D U-Net, HP2, Data",
    "3D U-Net, padding, Data", "3D U-Net, padding, Robust",
    "3D U-Net, shrinking, Data", "2D U-Net, Robust, Data"
],
        list(range(3))]
idxs = pd.MultiIndex.from_product(idxs, names=["Model", "i"])
data_res.index = idxs
data_res = data_res.rename(columns=lambda x: x.capitalize()
                           if "fg_" not in x else x[3:].capitalize(), )
# %%
files = glob.glob(
    "/homes/pmcd/Peter_Patrick3/*data*/predictions/csv/results.csv")
files.sort()
full = []
for i in files:
    df = pd.read_csv(i)[["MJ", "precision", "recall"]].rename(columns={"MJ":"Dice"})
    full.append(pd.concat({i: df}, names=['Model']))
mpu_res = pd.concat(full).groupby(["Model"]).agg("mean")

idxs = [["MPUNet, no D.A., Data", "MPUNet, Data"], list(range(3))]
idxs = pd.MultiIndex.from_product(idxs, names=["Model", "i"])
mpu_res.index = idxs
mpu_res = mpu_res.rename(columns=lambda x: x.capitalize()
                         if "fg_" not in x else x[3:].capitalize(), )
mpu_res = pd.concat([mpu_res, data_res])
mpu_res.groupby(['Model']).agg(['mean', 'std'
                                 ]).apply(bold_max).to_latex(**latex_kwargs,
                                             buf="./tables/new-models.tex",
                                             caption="",
                                             label="tab:a-a")