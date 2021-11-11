# Ace2_expression_neuts

In this project we want to understand how Ace2 expression on cell surface affects neutralization of SARS-CoV-2 virus by human sera.  
  
We have received Ace2 expressing HEK-293T cells from Kenneth Matreyek. We have several clones of these cells (consensus_Kozak, G, C, A), which have [differential Ace2 surface expression](https://journals.plos.org/plospathogens/article?id=10.1371/journal.ppat.1009715) due to modifications in Kozak sequence upstream of Ace gene. 

In these experiments we use Wuhan-1+614G luciferase reporter virus.  

The experimental steps are as follows:   
- test ability of this virus to infect different HEK-293T-Ace2 clones. Notebook: [virus_titers.ipynb](virus_titers.ipynb)
- perform mock and RBD-binding antibody depletion on human sera. Notebook: [rbd_ELISA.ipynb](rbd_ELISA.ipynb) 
- perform virus neutralization experiments with depleted and mock sera. Notebook: [virus_neutralization.ipynb](virus_neutralization.ipynb)  


Sera that is used in these experiments comes from Helen Chu's HAARVI study and is from individuals who have been infected with SARS-CoV-2 and subsequently received two doses of Phizer or Moderna vaccines. 

To run the analysis type:
	sbatch run_Hutch_cluster.bash

After the run is finished analysis results are found in [results](./results) folder

