import os
import sys
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import json

# Add the root directory to the Python path to enable importing from parent directory
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)
from meta import meta_func, meta_create

meta_create()
files_to_rename_txt = meta_func("rename_others_list", "your list file containing the file paths you want to rename")

with open(files_to_rename_txt) as file:
    files_to_rename = [line.rstrip() for line in file]

files_to_discard = []
valid_files = []

def check_bids_format(fn: str) -> bool:
    """
    Check if the file name meets the BIDS standard.
    """
    name = Path(fn).name
    pattern_bids = re.compile(
        r"^sub-[a-zA-Z0-9]+"
        r"(_[a-zA-Z0-9]+-[a-zA-Z0-9]+)*"
        r"_[a-zA-Z0-9]+"
        r"\.(nii\.gz|nii|json|tsv|bval|bvec|edf|set|fdt|mat)$"
    )
    match = pattern_bids.match(name)
    if not match:
        print(f"WARNING: {fn} does not follow the BIDS standard. File renaming will be skipped.")
    return bool(match)

def check_run_entity(fn: str) -> bool:
    """
    Check if the filename has a run-XX entity-label pair.
    """
    name = Path(fn).name
    has_entity = re.search(r"_run-\d{2}(?=(_|\.))", name)
    if bool(has_entity) ==False:
        print(f"WARNING: {fn} does not have a valid 'run-XX' entity-label pair. File renaming will be skipped.")
    return bool(has_entity)

def discard_t1_t2(fn: str) -> bool:
    """
    Check if the file is not T1 nor T2.
    """
    stem = Path(fn).name.removesuffix("".join(Path(fn).suffixes))
    not_anat = True
    if stem.endswith(("T1w", "T2w")):
        not_anat = False
        print(f"WARNING: {fn} is a T1 or T2 file, use rename_runs_qc_anat.py instead. File renaming will be skipped.")
    return not_anat

def discard_run00(fn: str) -> bool:
    """
    Discard 'run-00' files.
    """
    no_run00 = True
    if 'run-00' in fn:
        no_run00 = False
        print(f"WARNING: {fn} is a 'run-00' file, therefore it has already been renamed. File renaming will be skipped.")
    return no_run00

def discard_multiple_0x(valid_files: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Discard files if there is more than one run eligible to be renamed to run-01.
    """
    pattern = re.compile(r"(.*)_run-(\d{2})(.*)")
    groups = defaultdict(list)

    for f in valid_files:
        m = pattern.search(f)
        if not m:
            continue

        prefix, run, suffix = m.groups()
        key = (prefix, suffix)
        groups[key].append((f, int(run)))

    filtered_valid = []
    discarded = []

    for key, files in groups.items():
        runs = [run for _, run in files]
        non_01_runs = [run for run in runs if run != 1]

        if len(set(non_01_runs)) > 1:
            for f, _ in files:
                discarded.append(
                    (f, "More than one run different from run-01")
                )
                print(f"WARNING: {f} can't be renamed to 'run-01' as there are multiple runs in the list. File renaming will be skipped.")
            continue

        for f, _ in files:
            filtered_valid.append(f)
    
    return filtered_valid, discarded

def find_sidecars(valid_files: list[str]) -> list[str]:
    """
    Given a list of files, return a new list including all sidecars
    that share the same stem (ignoring extensions).
    """
    original_set = set(valid_files)
    expanded_set = set(valid_files)

    stem_groups = defaultdict(list)
    for fn in valid_files:
        path = Path(fn)
        folder = path.parent
        stem = path.name.removesuffix("".join(path.suffixes))
        stem_groups[(folder, stem)].append(fn)

    for (folder, stem), _ in stem_groups.items():
        for sibling in folder.iterdir():
            if not sibling.is_file():
                continue
            sibling_stem = sibling.name.removesuffix("".join(sibling.suffixes))
            if sibling_stem == stem:
                sibling_str = str(sibling)
                if sibling_str not in original_set:
                    print(f"\n{sibling_str} will also be renamed.")
                    expanded_set.add(sibling_str)
                
    return sorted(expanded_set)

def rename_01_to_00(fn:str):
    """
    Rename a 'run-01' files to 'run-00'.
    """
    path = Path(fn)
    stem = path.name
    if re.search(r"_run-01(?=(_|\.))", stem):
        new_name = stem.replace("run-01", "run-00")
        new_path = path.with_name(new_name)
        new_fn = str(new_path)
        print(f"\nTrying to rename {fn} -> {new_fn}")
        if new_path.exists():
            print(f"WARNING: {new_fn} already exists, therefore the renaming was already done. File renaming will be skipped.")
            return None   
        else:
            try:
                path.rename(new_path)
                print(f"{fn} -> {new_fn} Successfully renamed!")
                return new_fn
            except Exception as e:
                print(f"ERROR: Could not rename {fn} -> {new_fn}. Reason: {e}")
                return None

def rename_0x_to_01(fn:str):
    """
    Rename a 'run-0x' (not run-00, not run-01) files to 'run-01'.
    """
    path = Path(fn)
    stem = path.name
    match = re.search(r"run-(\d{2})(?=(_|\.))", stem)

    if match and match.group(1) not in ("00", "01"):
        old_run = match.group(0)
        new_name = stem.replace(old_run, "run-01")
        new_path = path.with_name(new_name)
        new_fn = str(new_path)
        print(f"\nTrying to rename {fn} -> {new_fn}")
        if new_path.exists():
            print(f"WARNING: {new_fn} still exists, so you will to also add it to the list. File renaming will be skipped.")
            return None
        
        try:
            path.rename(new_path)
            print(f"{fn} -> {new_fn} Successfully renamed!")
            return new_fn
        except Exception as e:
            print(f"ERROR: Could not rename {fn} -> {new_fn}. Reason: {e}")
            return None

def modify_scans_tsv(fn:str, new_fn:str):
    path = Path(fn)
    path_new = Path(new_fn)
    csv_fn = f"{path.parent.name}/{path.name}"
    csv_fn_new = f"{path_new.parent.name}/{path_new.name}"
    scans_files = list(Path(path).parent.parent.glob('*scans.tsv'))
    if not scans_files:
        print("scans.tsv file was not found.")
        return
    scans_file = scans_files[0]
    scans_df = pd.read_csv(scans_file, sep="\t")
    scans_df.loc[scans_df['filename'] == csv_fn, 'filename'] = csv_fn_new
    scans_df.to_csv(scans_file, sep="\t", index=False)
    print(f"{csv_fn} modified to {csv_fn_new} in 'filename' column of {scans_file}")

def modify_intendedfor(fn:str, new_fn:str):
    path = Path(fn)
    new_path = Path(new_fn)
    def get_path_after_sub(full_path):
        parts = full_path.parts
        for i, part in enumerate(parts):
            if part.startswith("sub-"):
                rel_path = Path(*parts[i+1:])
                return str(rel_path)
            else:
                continue
        print(f"WARNING: {full_path} is not in a BIDS folder.")
        return None
    fn_aftersub = get_path_after_sub(path)
    new_fn_aftersub = get_path_after_sub(new_path)

    if path.parent.name in ('dwi', 'func'):
        print("\nUpdating the fieldmaps metadata...")
        fmap_folder = path.parent.parent / "fmap"
        fmap_jsons = sorted(fmap_folder.glob("*.json"))
        if not fmap_jsons:
            print(f"WARNING: No JSON fieldmap files found in {fmap_folder}")

        for json_path in fmap_jsons:
            with open(json_path, "r") as f:
                fmap_data = json.load(f)
            intended = fmap_data.get("IntendedFor", None)
            if intended is None:
                continue
            if fn_aftersub in intended:
                fmap_data["IntendedFor"] = [
                    new_fn_aftersub if x == fn_aftersub else x for x in intended
                ]
                with open(json_path, "w") as f:
                    json.dump(fmap_data, f, indent=4)
                print(f"{fn_aftersub} modified to {new_fn_aftersub} in 'IntendedFor' field of {json_path}")
            else:
                continue
    
    if fn_aftersub is None or new_fn_aftersub is None:
        return

for file in files_to_rename:
    print(f"\nNow processing file {file}")

    print("\nChecking if the file follows the BIDS file structure...")
    if not check_bids_format(file):
        files_to_discard.append((file, "Invalid BIDS format"))
        continue

    print("\nChecking if the file has a 'run' entity...")
    if not check_run_entity(file):
        files_to_discard.append((file, "No run entity"))
        continue

    print("\nChecking if the file is not T1 nor T2...")
    if not discard_t1_t2(file):
        files_to_discard.append((file, "T1/T2 excluded"))
        continue

    print("\nChecking if the file has not been renamed before...")
    if not discard_run00(file):
        files_to_discard.append((file, "Already run-00"))
        continue

    valid_files.append(file)

print("\nChecking if the list does not contain several runs other than run-01...")
valid_files, multi_discards = discard_multiple_0x(valid_files)
files_to_discard.extend(multi_discards)

print("\nAdding image and sidecar files to the renaming list...")
valid_files = find_sidecars(valid_files)

for file in valid_files:
    path = Path(file)
    stem = path.name

    if re.search(r"_run-01(?=(_|\.))", stem):
        new_file = rename_01_to_00(file)
    else:
        new_file = rename_0x_to_01(file)

    if new_file is None:
        continue

    suffixes = ''.join(path.suffixes)
    if suffixes == ".nii.gz":
        print("\nUpdating the scans.tsv file...")
        modify_scans_tsv(file, new_file)

        modify_intendedfor(file, new_file)
