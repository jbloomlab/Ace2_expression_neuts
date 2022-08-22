
"""``snakemake`` file that runs entire analysis."""

# Imports ---------------------------------------------------------------------
import glob
import itertools
import os.path
import os
import textwrap
import urllib.request

import Bio.SeqIO

import pandas as pd

# Configuration  --------------------------------------------------------------

configfile: 'config.yaml'

# Functions -------------------------------------------------------------------

def nb_markdown(nb):
    """Return path to Markdown results of notebook `nb`."""
    return os.path.join(config['summary_dir'],
                        os.path.basename(os.path.splitext(nb)[0]) + '.md')

# Target rules ---------------------------------------------------------------

localrules: all

rule all:
    input:
        'results/summary/virus_titers.md',
        'results/summary/rbd_depletions.md',
        'results/summary/virus_neutralization.md',
        'results/summary/virus_neutralization_mAbs.md',
        'results/summary/ACE2_expression_vs_infectivity.md'
        


# Rules ---------------------------------------------------------------------

rule get_virus_titers:
    """calculate virus titers"""
    input:
        config['virus_titers']
    output:
        nb_markdown=nb_markdown('virus_titers.ipynb')
    params:
        nb='virus_titers.ipynb'
    shell:
        "python scripts/run_nb.py {params.nb} {output.nb_markdown}"

rule get_RBD_depletions:
    """plot RBD depletion data"""
    input:
        elisa_pv_1 = config['elisa_pseudovirus_dil1'],
        elisa_pv_2 = config['elisa_pseudovirus_dil2'],
        
    output:
        nb_markdown=nb_markdown('rbd_depletions.ipynb')
    params:
        nb='rbd_depletions.ipynb'
    shell:
        "python scripts/run_nb.py {params.nb} {output.nb_markdown}"
        
rule plot_neuts:
    """plot neut curves"""
    input:
        sample_info=config['sample_information'],
        depletion_neuts=config['depletion_neuts']
    output:
        nb_markdown=nb_markdown('virus_neutralization.ipynb')
    params:
        nb='virus_neutralization.ipynb'
    shell:
        "python scripts/run_nb.py {params.nb} {output.nb_markdown}"

rule plot_mAb_neuts:
    """plot neut curves"""
    input:
        depletion_neuts=config['mAb_neuts']
    output:
        nb_markdown=nb_markdown('virus_neutralization_mAbs.ipynb')
    params:
        nb='virus_neutralization_mAbs.ipynb'
    shell:
        "python scripts/run_nb.py {params.nb} {output.nb_markdown}"

rule plot_RLU_vs_ACE2:
    """plot infection vs ACE2 expression"""
    input:
        depletion_neuts=config['ACE2_expression_df']
    output:
        nb_markdown=nb_markdown('ACE2_expression_vs_infectivity.ipynb')
    params:
        nb='ACE2_expression_vs_infectivity.ipynb'
    shell:
        "python scripts/run_nb.py {params.nb} {output.nb_markdown}"

