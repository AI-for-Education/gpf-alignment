import pandas as pd
from pathlib import Path
import zipfile
import os
import tempfile
from lxml import etree
from PIL import Image
from openpyxl import load_workbook
from io import BytesIO


def get_namespaces_from_xml(xml_path, required_prefixes=None):
    """Extract namespaces from an XML file with fallbacks for required prefixes."""
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Standard fallback namespaces
    fallback_namespaces = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    namespaces = {}

    # Get all namespaces from the root element
    nsmap = root.nsmap

    # Map default namespace to 'main'
    if None in nsmap:
        namespaces["main"] = nsmap[None]

    # Add other namespaces
    for prefix, uri in nsmap.items():
        if prefix is not None:
            namespaces[prefix] = uri

    # Ensure required prefixes are present, use fallbacks if needed
    if required_prefixes:
        for prefix in required_prefixes:
            if prefix not in namespaces and prefix in fallback_namespaces:
                namespaces[prefix] = fallback_namespaces[prefix]
                print(f"Warning: Using fallback namespace for {prefix}")

    # for prefix, uri in namespaces.items():
    #     print(f"  {prefix}: {uri}")

    return namespaces


def debug_xml(rv_path):
    # Parse the XML
    rv_tree = etree.parse(rv_path)
    rv_root = rv_tree.getroot()

    # Print the raw XML to see exactly what we're dealing with
    print("Raw XML:")
    print(etree.tostring(rv_root, pretty_print=True).decode())

    # Print all namespaces in use
    print("\nNamespaces:")
    for prefix, uri in rv_root.nsmap.items():
        print(f"Prefix: {prefix}, URI: {uri}")

    # Print all children and their attributes
    print("\nChildren and attributes:")
    for child in rv_root:
        print(f"Tag: {child.tag}")
        print(f"Attributes: {child.attrib}")

    # Try different XPath expressions
    namespaces = {
        "rvr": "http://schemas.microsoft.com/office/spreadsheetml/2022/richvaluerel",
        "r": "http://purl.oclc.org/ooxml/officeDocument/relationships",
    }

    print("\nTrying different XPath expressions:")
    print("1. //rel:", len(rv_root.xpath("//rel")))
    print("2. //*[local-name()='rel']:", len(rv_root.xpath("//*[local-name()='rel']")))
    print("3. With namespace:", len(rv_root.xpath("//rvr:rel", namespaces=namespaces)))

    # Try getting attributes directly
    print("\nTrying to get attributes directly:")
    for elem in rv_root.xpath("//*[local-name()='rel']"):
        print(f"Element attributes: {elem.attrib}")

    return rv_root


def load_excel_with_richval_images(excel_path, header_row=1):
    """
    Load an Excel file with rich value images to a pandas DataFrame with image references.

    Args:
        excel_path (str): Path to the Excel file
        header_row (int): Row to use as header (0-based index)

    Returns:
        tuple: (DataFrame with image references, dict mapping cell positions to image paths)
    """
    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(excel_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        df = pd.read_excel(excel_path, header=header_row)
        df["Original Item"] = df["Original Item"].astype("string")

        image_mapping = {}
        # Parse the rich value relationships with lxml
        rels_path = os.path.join(
            temp_dir, "xl", "richData", "_rels", "richValueRel.xml.rels"
        )
        rels_ns = get_namespaces_from_xml(rels_path)
        rels_tree = etree.parse(rels_path)
        rels_root = rels_tree.getroot()

        rid_to_image = {}
        for rel in rels_root.xpath(".//main:Relationship", namespaces=rels_ns):
            rid = rel.get("Id")
            target = rel.get("Target")
            image_path = os.path.join(temp_dir, "xl", "richData", *target.split("/"))
            rid_to_image[rid] = image_path

        # Parse rich value mappings
        rv_path = os.path.join(temp_dir, "xl", "richData", "richValueRel.xml")
        index_to_rid = {}
        rv_ns = get_namespaces_from_xml(rv_path, required_prefixes=["r"])
        rv_tree = etree.parse(rv_path)
        rv_root = rv_tree.getroot()
        for i, rel in enumerate(rv_root):
            rid = rel.attrib[f'{{{rv_ns["r"]}}}id']
            index_to_rid[i] = rid
            # print(f"Found relationship {i}: {rid}")

        # Parse worksheet to find cells with images
        sheet_path = os.path.join(temp_dir, "xl", "worksheets", "sheet1.xml")
        sheet_ns = get_namespaces_from_xml(sheet_path)
        sheet_tree = etree.parse(sheet_path)
        sheet_root = sheet_tree.getroot()

        # Find all cells with vm attribute
        for cell in sheet_root.xpath(".//main:c[@vm]", namespaces=sheet_ns):
            # Get cell reference and vm value
            cell_ref = cell.get("r")
            vm = int(cell.get("vm"))
            # print((cell_ref, vm))

            # Convert vm to index (0-based)
            index = vm - 1

            # Get corresponding rId and image path
            if index in index_to_rid:
                rid = index_to_rid[index]
                if rid in rid_to_image:
                    # Convert Excel cell reference to row/col index
                    col = "".join(filter(str.isalpha, cell_ref))
                    excel_row = (
                        int("".join(filter(str.isdigit, cell_ref))) - 1
                    )  # 0-based Excel row
                    col_idx = (
                        sum(
                            (ord(c) - ord("A") + 1) * (26**i)
                            for i, c in enumerate(reversed(col))
                        )
                        - 1
                    )

                    # Adjust row index for DataFrame (subtract header rows)
                    df_row = excel_row - (header_row + 1)

                    # Only process if the row is within the DataFrame
                    if df_row >= 0:
                        # Load the image using Pillow
                        image = Image.open(rid_to_image[rid])

                        # Store the Pillow image object in the mapping
                        image_mapping[(df_row, col_idx)] = image

                        # Update DataFrame with image reference
                        df.iloc[df_row, col_idx] = (
                            f"[Image: {os.path.basename(rid_to_image[rid])}]"
                        )

    return df, image_mapping


def load_excel_with_drawing_images(file, header_row=1):
    # Load your workbook and sheet as you want, for example
    wb = load_workbook(file)
    sheet = wb["Sheet1"]

    df = pd.read_excel(file, header=header_row)
    df["Original Item"] = df["Original Item"].astype("string")

    image_mapping = {}
    for openpyxl_image in sheet._images:
        col = openpyxl_image.anchor._from.col
        row = openpyxl_image.anchor._from.row
        df_row = row - (header_row + 1)
        image = Image.open(BytesIO(openpyxl_image._data()))
        image_mapping[(df_row, col)] = image

    return df, image_mapping
