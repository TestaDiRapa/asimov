from bs4 import BeautifulSoup

STATUS_CODE = {
    "Negative": '-',
    "Positive": '+',
    "Equivocal": '?',
    "Indeterminate": '?'
}


def find_receptor_status(soup, selector):
    tag = soup.find(name=selector)
    if tag is None or tag["procurement_status"] != "Completed":
        return '?'
    return STATUS_CODE[tag.text]


def identify_subtype(clinical_filename):
    with open(clinical_filename, 'r', encoding="utf-8") as input_file:
        raw_xml = input_file.read()

    soup = BeautifulSoup(raw_xml, 'lxml')

    her2 = find_receptor_status(soup, "brca_shared:lab_procedure_her2_neu_in_situ_hybrid_outcome_type")
    if her2 == '?':
        her2 = find_receptor_status(soup, "brca_shared:lab_proc_her2_neu_immunohistochemistry_receptor_status")

    er = find_receptor_status(soup, "brca_shared:breast_carcinoma_estrogen_receptor_status")

    pr = find_receptor_status(soup, "brca_shared:metastatic_breast_carcinoma_progesterone_receptor_status")

    if (er == '+' or pr == '+') and her2 == '-':
        return "LuminalA"

    elif (er == '+' or pr == '+') and her2 == '+':
        return "LuminalB"

    elif (er == '-' or pr == '-') and her2 == '-':
        return "TNBS"

    elif (er == '-' or pr == '-') and her2 == '+':
        return "HER2+"

    else:
        return "Unclear"
