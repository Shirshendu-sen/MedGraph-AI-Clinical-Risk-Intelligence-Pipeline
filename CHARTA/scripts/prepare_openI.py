"""Prepare OpenI dataset: parse NLM XML reports, write individual .txt files, copy images."""

# OpenI reports are NLM XML format, NOT BioC XML
# Use lxml to parse the AbstractText elements with Label attributes
from lxml import etree
import pathlib, shutil, argparse

SECTIONS = ["COMPARISON", "INDICATION", "FINDINGS", "IMPRESSION"]


def parse_openI_xml_report(xml_path: str) -> dict:
    tree = etree.parse(xml_path)
    uid  = tree.find(".//uId").get("id", pathlib.Path(xml_path).stem)
    result = {"uid": uid}
    for section in SECTIONS:
        elem = tree.find(f".//AbstractText[@Label='{section}']")
        result[section.lower()] = (elem.text or "").strip() if elem is not None else ""
    return result


def write_report_as_text(report: dict, output_dir: str) -> None:
    lines = []
    for section in SECTIONS:
        text = report.get(section.lower(), "")
        if text:
            lines.append(f"{section}:\n{text}")
    pathlib.Path(output_dir, f"openI_{report['uid']}.txt").write_text(
        "\n\n".join(lines), encoding="utf-8", errors="replace")


def run(input_dir, txt_output, img_output):
    pathlib.Path(txt_output).mkdir(parents=True, exist_ok=True)
    pathlib.Path(img_output).mkdir(parents=True, exist_ok=True)
    for xml_path in pathlib.Path(input_dir).glob("*.xml"):
        report = parse_openI_xml_report(str(xml_path))
        write_report_as_text(report, txt_output)
    for png in pathlib.Path(input_dir).glob("*.png"):
        shutil.copy(png, img_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare OpenI .txt files and images")
    parser.add_argument("--input", default="data/openI/")
    parser.add_argument("--txt_output", default="data/raw/txt/openI/")
    parser.add_argument("--img_output", default="data/raw/images/openI/")
    args = parser.parse_args()

    run(args.input, args.txt_output, args.img_output)
    print(f"Processed OpenI data from {args.input}")
    print(f"  Text files  -> {args.txt_output}")
    print(f"  Images      -> {args.img_output}")
