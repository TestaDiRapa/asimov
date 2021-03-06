from bs4 import BeautifulSoup
from dataset import BarcodeFinder, folder_generator
import os
import pandas as pd
import pickle
import requests


def __get_mature_mirnas(pre_mirna):
    url = "http://carolina.imis.athena-innovation.gr/diana_tools/web/index.php?r=tarbasev8/auto-complete-mirnas&" \
          "expansion=both&max_num=5&term={}"
    r = requests.get(url.format(pre_mirna), timeout=10)
    return r.json()


def __find_num_pages(mirna):
    url = "http://carolina.imis.athena-innovation.gr/diana_tools/web/index.php?r=tarbasev8/index&miRNAs[]={}"
    r = requests.get(url.format(mirna), timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    if len(soup.select("ul.pagination li")) == 0:
        return 1
    else:
        return len(soup.select("ul.pagination li")) - 2


def __get_genes_data(mirna, page):
    genes = list()
    url = "http://carolina.imis.athena-innovation.gr/diana_tools/web/index.php?r=tarbasev8/index&" \
          "miRNAs[]={}&page={}"
    r = requests.get(url.format(mirna, page), timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for row in soup.select("tr.first-level"):
        if len(row.select("a[data-target^='#ENST']")) > 0:
            id_ = row.select("a[data-target^='#ENST']")[0]
            gene_container = soup.select("div{} div.modal-body div.row:nth-child(4) > div.col-md-5:nth-child(2)"
                                         .format(id_["data-target"]))
            name_container = soup.select("div{} div.modal-body div.row:nth-child(5) > div.col-md-5:nth-child(2)"
                                         .format(id_["data-target"]))
            if len(gene_container) > 0 and len(name_container) > 0:
                score_tag = row.select("a[href^='http://diana.imis.athena-innovation.gr']")
                if len(score_tag) > 0:
                    score = float(score_tag[0].text.strip())
                else:
                    score = 0
                genes.append((gene_container[0].text.strip(), name_container[0].text.strip(), score))
    return genes


def mirna_genes_interaction(pre_mirna):
    mirnas = __get_mature_mirnas(pre_mirna)
    genes = list()
    for mirna in mirnas:
        num_pages = __find_num_pages(mirna)
        for page in range(1, num_pages+1):
            genes += __get_genes_data(mirna, page)
    return genes


def get_interactions_over_threshold(threshold, gene_name, interaction_file="../data/mirna_genes_interaction.pkl"):
    mirna_interactions = pickle.load(open(interaction_file, "rb"))
    final_genes = set()
    for mirna, interactions in mirna_interactions.items():
        for gene in interactions:
            if gene[2] >= threshold:
                final_genes.add(gene[gene_name])

    return list(final_genes)
