# FlowCyt Data

This repository contains the benchmark dataset and code for the paper "FlowCyt: A Comparative Study of Deep Learning Approaches for Multi-Class Classification in Flow Cytometry" accepted at [CHIL@2024](https://chilconference.org/), [Track 2: Applications And Practice](https://chilconference.org/call-for-papers.html#tab-track-2) for benchmarking deep learning models in Medicine.

FlowCyt is the first comprehensive benchmark for evaluating multi-class single-cell classification methods on flow cytometry data. It comprises a richly annotated dataset of bone marrow samples from 30 patients, with ground truth labels for 5 important hematological cell types.

## Contents

- `raw/` : The original FCS (Flow Cytometry Standard) files output by the flow cytometer.
- `data_original/`: Contains CSV data for each sample, saved as *Case_{i}.csv* with all the six classes (A-population) of cells.
- `data_original_sub/`: Contains CSV sub-popoulation data for each sample, saved as *Case_{i}.csv* with all the five classes (sub-population) of cells.

Within the `raw/` directory all 30 patients are organized as this:
- `CaseN_A.fcs` : The full cell population (to be used for further analysis if keen on)
- `CaseN_O.fcs` : Only T lymphocytes 
- `CaseN_N.fcs` : Only B lymphocytes
- `CaseN_G.fcs` : Only monocytes
- `CaseN_P.fcs` : Only mast cells 
- `CaseN_K.fcs` : Only HSPCs
- `CaseN_B.fcs` : Other cells

Each FCS file contains measurements for 200,000 - 1,000,000 cells. The data has 14 dimensions, however 'TIME', 'SS PEAK', and 'SS TOF (time of flight)' don't contain any biological information and are due to the cytometer machine itself. Therefore, the ones to be used to reproduce all the experiments in the paper are only 12 of them: forward/side scatter and 10 antibody markers. However, researchers are free to use them all.

## data_original (*Case_{i}.csv*)

This CSV file contains the total cell population (A-population) for each patient sample, where the ground truth cell type labels are encoded within the last column called 'label'. This enables large supervised learning experiments.

## data_original_sub (*Case_{i}.csv*)

This CSV file contains the sub cell population (sub-population) for each patient sample, where the ground truth cell type labels are encoded within the last column called 'label'. This enables supervised learning experiments.

## Quick start with the graph lists
For a quick start, we include the two normalized graph lists to be used for `inductive/` and `transductive/` experiments, each of them created by kNN with $k=7$, both for A-populations `A_graph.pt` and sub-populations `sub_graph.pt`of cells per each patient.
To reproduce the graphs:
```python
python -u -m data.graph_generation --in_dir data/data_original --out_dir data/A_graph.pt
python -u -m data.graph_generation --in_dir data/data_original_sub --out_dir data/sub_graph.pt
````

## Usage 
To recreate the CSV data for total population and save them as  `data_original/`:
```python
python A_generation.py
````

To create the CSV data only for sub-population (${\text{O},\text{N},\text{G},\text{P},\text{K}}$) and save them as  `data_original_sub/`:
```python
python sub_generation.py
````

## Citation
If you find this benchmark dataset useful in your research, please cite the following paper:

```bash
@inproceedings{flowcyt2024,
  title={FlowCyt: A Comparative Study of Deep Learning Approaches for Multi-Class Classification in Flow Cytometry},
  author={Bini, Lorenzo and Mojarrad, Fatemeh Nassajian and Liarou, Margarita and Matthes, Thomas and Marchand-Maillet, Stéphane},
  booktitle={Conference on Health, Inference, and Learning (CHIL)},
  year={2024}
}
```

## Contact
Please contact the authors with any questions or feedback about the benchmark. We are happy to receive suggestions for extensions and collaborations!
