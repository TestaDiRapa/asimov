import requests


def enrichr_query(gene_list, library):
    enrichr_add_list_url = 'http://maayanlab.cloud/Enrichr/addList'
    enrichr_enrich_url = 'http://maayanlab.cloud/Enrichr/enrich'
    genes_str = '\n'.join(gene_list)
    payload = {
        'list': (None, genes_str),
        'description': (None, "Gene list")
    }
    add_list_response = requests.post(enrichr_add_list_url, files=payload)
    if not add_list_response.ok:
        raise Exception("Error analyzing gene list: {}".format(add_list_response.text))

    query_string = '?userListId=%s&backgroundType=%s'
    user_list_id = add_list_response.json()["userListId"]
    response = requests.get(
        enrichr_enrich_url + query_string % (user_list_id, library)
    )
    if not response.ok:
        raise Exception('Error fetching enrichment results')

    return response.json()
