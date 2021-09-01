import glob
import dicom2nifti.convert_dir as conv_dir
import os
import shutil
import json
os.chdir("/homes/pmcd/Peter_Patrick3")
base_dir = "/homes/pmcd/Peter_Patrick3"
train_split = 0.7
chances = {
    "train": train_split * 0.6,
    "val": train_split * 0.2,
    "test": train_split * 0.2,
    "true_test": 1 - train_split
}
#%%
examples = [
    folder for folder in glob.glob(f"{base_dir}/all/*")
    if os.path.exists(f"{folder}/rest.nii.gz") and
    os.path.exists(f"{folder}/annotations.nii.gz")
]
data = {}
sum_val = 0
for key, value in chances.items():
    end = sum_val + (len(examples) * value)
    first_idx = int(sum_val)
    end_idx = round(end)
    print(end, end_idx, first_idx, sum_val)
    sum_val = end_idx
    data[key] = examples[first_idx:end_idx]
data[key] = examples[first_idx:]
print(len([j for i in data.values() for j in i]))
print({key: (val[0],val[-1]) for key,val in data.items()})
#%%
idx = 0
for ds_name, folders in data.items():
    ds_base_folder = f"/homes/pmcd/Peter_Patrick3/{ds_name}"
    try:
        os.mkdir(ds_base_folder)
    except:
        pass

    for folder in folders:
        file_name = folder.split("/")[-1]
        try:
            os.mkdir(f"{ds_base_folder}/images")
            os.mkdir(f"{ds_base_folder}/labels")
        except:
            pass
        if os.path.exists(f"{folder}/rest.nii.gz"
                         ) and os.path.exists(f"{folder}/annotations.nii.gz"):
            os.symlink(
                f"{folder}/rest.nii.gz",
                f"{base_dir}/{ds_name}/images/{file_name}.nii.gz"
            )
            os.symlink(
                f"{folder}/annotations.nii.gz",
                f"{base_dir}/{ds_name}/labels/{file_name}.nii.gz"
            )
