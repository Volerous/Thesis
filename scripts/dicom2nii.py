#%%
import glob
import dicom2nifti.convert_dir as conv_dir
import os
import shutil
import json
from pydicom.filereader import read_dicomdir
from pydicom import dcmread
import pydicom
from pydicom.fileset import FileSet
os.chdir("/homes/pmcd/Peter_Patrick3")
base_dir = "/homes/pmcd/Peter_Patrick3"

#%%
folders = glob.glob("/homes/pmcd/Peter_Patrick3/DICOM/*/*")
try:
    os.mkdir("/homes/pmcd/Peter_Patrick3/all")
except:
    pass
ds = read_dicomdir("/homes/pmcd/Peter_Patrick3/DICOMDIR")
#%%
lst = []

#%%
for idx,i in enumerate(ds.patient_records):
    if idx <= 60:
        for j in i.children:
            for k in j.children:
                if "REST" in k[0x0008103e].value:
                    for l in k.children:
                        # print(l[0x00041500].value)
                        print(i[0x00100040].value)
                        break

#%%

# for idx, folder in enumerate(folders):
#     # (0004, 1500)
#     print()
# file_name = f"anon{idx}"
# print(file_name)
# out_dir = base_dir + "/all/" + file_name
# nii_file = glob.glob(f"{folder}/*.nii.gz")
# completed = {
#     "rest": False,
#     "stress": False,
#     "ct": False,
#     "original": folder
# }
# print(nii_file)
# anon_file = f"{out_dir}/annotations.nii.gz"
# try:
#     os.mkdir(out_dir)
# except:
#     pass

# try:
#     shutil.copy(nii_file[0], anon_file)
# except:
#     pass
# # conv_dir.convert_directory(folder, out_dir, True, False)
# for i in glob.glob(f"{out_dir}/*"):
#     if "rest" in i.lower():
#         os.rename(i, f"{out_dir}/rest.nii.gz")
#         completed["rest"] = True
#     elif "stress" in i.lower():
#         os.rename(i, f"{out_dir}/stress.nii.gz")
#         completed["stress"] = True
#     elif "ct" in i.lower():
#         os.rename(i, f"{out_dir}/ct.nii.gz")
#         completed["ct"] = True

# with open(f'{out_dir}/data.json', 'w') as outfile:
#     json.dump(completed, outfile)
# break
#%%
