from Bio import Entrez
import json

# Set your email (required by PubMed)
Entrez.email = "bsantisi@gmail.com"

# Search for papers
handle = Entrez.esearch(db="pubmed", term="CRISPR", retmax=10000)
record = Entrez.read(handle)
ids = record["IdList"]

# Fetch abstracts
handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
abstracts = Entrez.read(handle)

# Save to file
with open("pubmed_abstracts.json", "w") as f:
    json.dump(abstracts, f)