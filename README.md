# ACE2 expression neuts

In this project we want to understand how Ace2 expression on target cell surface affects neutralization of SARS-CoV-2 virus by human sera and monoclonal antibodies.

We are using ACE2 expressing HEK-293T from Kenneth Matreyek. We have several clones of these cells ('high', 'medium', 'low', 'very low'), which have [differential Ace2 surface expression](https://journals.plos.org/plospathogens/article?id=10.1371/journal.ppat.1009715) due to modifications in Kozak sequence upstream of ACE2 gene.

In these experiments we use Wu-1-Hu-614G virus.

The experimental steps are as follows:
- test ability of this virus to infect different HEK-293T-Ace2 clones. Notebook: [virus_titers.ipynb](virus_titers.ipynb)
- test ACE2 expression in different cell clones. Notebook: [ACE2_expression_vs_infectivity.ipynb](ACE2_expression_vs_infectivity.ipynb)
- perform mock and RBD-binding antibody depletion on human sera. Notebook: [rbd_depletions.ipynb](rbd_depletions.ipynb)
- perform virus neutralization experiments with RBD-binding antibody depleted and nondepleted sera. Notebook: [virus_neutralization.ipynb](virus_neutralization.ipynb)
- perform virus neutralization experiments monoclonal antibodies targetting different Spike epitopes. Notebook: [virus_neutralization_mAbs.ipynb](virus_neutralization_mAbs.ipynb)


Sera that is used in these experiments comes from HAARVI study and is from individuals who have been infected with SARS-CoV-2 and subsequently received a two-dose vaccination.

To run the analysis using snakemake pipeline type:

```
sbatch run_Hutch_cluster.bash
```

After the run is finished analysis results are found in [results](./results) folder.
