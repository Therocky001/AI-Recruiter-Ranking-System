import zipfile
import xml.etree.ElementTree as ET
import sys
import os

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path, 'r') as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = []
            for p in tree.findall('.//w:p', ns):
                texts = [node.text for node in p.findall('.//w:t', ns) if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n".join(paragraphs)
    except Exception as e:
        return f"Error reading {docx_path}: {e}"

if __name__ == "__main__":
    files = [
        r"d:\Hackathons\indiaruns data &  Ai challanges\Data\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\job_description.docx",
        r"d:\Hackathons\indiaruns data &  Ai challanges\Data\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\redrob_signals_doc.docx",
        r"d:\Hackathons\indiaruns data &  Ai challanges\Data\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\submission_spec.docx"
    ]
    for f in files:
        text = extract_text_from_docx(f)
        out_name = os.path.basename(f) + ".txt"
        with open(out_name, "w", encoding="utf-8") as out:
            out.write(text)
