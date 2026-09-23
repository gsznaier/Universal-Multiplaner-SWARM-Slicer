# Universal Multiplanar SWARM slicer

## Prerequisites 

This project uses [conda](https://docs.conda.io/) to manage its Python environment. You'll need one of the following installed: 

- **[Miniconda](https://docs.anaconda.com/miniconda/)** (recommended, lightweight) 
- **[Anaconda](https://www.anaconda.com/download)** (full distribution with many preinstalled packages

You'll also need [Git](https://git-scm.com/downloads) to clone the repository. 

To check that conda is installed and available in your terminal, run: 

```bash conda --version ``` 

> **Windows users:** Run the commands below in **Anaconda Prompt** (installed with Anaconda/Miniconda), or run `conda init` once so that `conda` works in PowerShell or Command Prompt. 

## Installation

To install the software dependencies, run the following commands in your terminal:

```bash
# Clone the repository
git clone https://github.com/gsznaier/Universal-Multiplaner-SWARM-Slicer.git
cd Universal-Multiplaner-SWARM-Slicer

# Create and activate conda environment
conda env create -f environment.yml
conda activate slicer

# Install local package in editable mode
pip install -e .
```
The installation can take 5-10 minutes on a normal computer.
---

## Run instructions

Execute the slicer by passing parameters directly through the command line:

```bash
cd slicer
python main.py --job_names cube --stl_names ../dataset/cube.stl --printer_profiles ../printer_profiles/1_printer_profile_cylindrical_contour_1mm.json
```

### Concentric tube robot (CTR) profiles
To scale execution to multiple CTR printers, replace the `--printer_profiles` argument with one of the pre-configured system profiles:

| CTR Count | Profile Path |
| :--- | :--- |
| **1 CTR** | `../printer_profiles/1_printer_profile_cylindrical_contour_1mm.json` |
| **2 CTRs** | `../printer_profiles/2_printer_profile_cylindrical_contour_1mm.json` |
| **4 CTRs** | `../printer_profiles/4_printer_profile_cylindrical_contour_1mm.json` |
| **6 CTRs** | `../printer_profiles/6_printer_profile_cylindrical_contour_1mm.json` |
| **8 CTRs** | `../printer_profiles/8_printer_profile_cylindrical_contour_1mm.json` |
| **14 CTRs** | `../printer_profiles/14_printer_profile_cylindrical_contour_1mm.json` |

---

## Visualizing print paths

Generated print paths can be viewed interactively with `plot_paths.py`.

```bash
python plot_paths.py --job_names <job_name> --printer_profiles ../printer_profiles/<N>_printer_profile_<profile_type>.json --load_data_path ../results/shells
```

| Placeholder | Description | Example |
|---|---|---|
| `<job_name>` | Shape/job to visualize | `benchy`, `bunny`, `duck`, `heart` |
| `<N>` | Number of printheads (selects the `<job_name>_<N>_printers_...` results folder) | `1`, `2`, `4`, `6`, `8`, `14` |
| `<profile_type>` | Remainder of the printer profile file name | `cylindrical_contour_1mm` |

**Example:** visualize the bunny printed with 4 printheads:

```bash
python plot_paths.py --job_names bunny --printer_profiles ../printer_profiles/6_printer_profile_cylindrical_contour_1mm.json --load_data_path ../results/shells
```

**Short form:**

```bash
python plot_paths.py <job_name> <N>
python plot_paths.py bunny 6
```

Running `python plot_paths.py` with no arguments lists all available jobs and prompts for one.

---

## Tested software versions

* **Operating systems:** Ubuntu 18.04.6 LTS, Windows 11 Education
* **Python:** 3.10
