import fitz  # PyMuPDF
import json
import base64

def parse_pdf_to_json(pdf_path, output_json_path):
    doc = fitz.open(pdf_path)
    document_data = {"pages": []}

    print(f"Parsing {pdf_path}...")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_dict = {
            "page_number": page_num + 1,
            "width": page.rect.width,
            "height": page.rect.height,
            "text_blocks": [], 
            "images": []
        }

        # Extract text spans while preserving PDF typography
        # ... [Inside parse_pdf_to_json, replace the text extraction loop:] ...

        # Extract text blocks while preserving PDF typography
        text_page = page.get_text("dict")

        for block in text_page.get("blocks", []):
            if block.get("type") != 0:
                continue

            block_text = ""
            font = "Arial"
            size = 8
            flags = 0
            color = 0

            first_span_found = False

            # PyMuPDF has already computed exactly where each line wraps within this
            # block's original column width -- that layout info is the source of truth. 
            line_records = []

            for line in block.get("lines", []):
                line_text = ""
                line_bbox = line.get("bbox")

                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text:
                        continue

                    # Capture formatting from the first valid span in the block
                    if not first_span_found:
                        font = span.get("font", "Arial")
                        size = span.get("size", 8)
                        flags = span.get("flags", 0)
                        color = span.get("color", 0)
                        first_span_found = True

                    # Add text (append space to separate spans cleanly)
                    line_text += text + " "

                line_text = line_text.strip()
                if not line_text:
                    continue

                block_text += line_text + "\n"

                if line_bbox:
                    line_records.append({
                        "bbox": [line_bbox[0], line_bbox[1], line_bbox[2], line_bbox[3]],
                        "text": line_text,
                    })

            block_text = block_text.strip()
            if not block_text:
                continue

            # Save the bounding box of the ENTIRE block
            bbox = block["bbox"]

            page_dict["text_blocks"].append({
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "text": block_text,
                "lines": line_records,
                "font": font,
                "size": size,
                "flags": flags,
                "color": color
            })

        # Extract Images & Bounding Boxes
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            rects = page.get_image_rects(xref)
            if rects:
                bbox = rects[0]
                base_image = doc.extract_image(xref)
                img_b64 = base64.b64encode(base_image["image"]).decode('utf-8')
                
                page_dict["images"].append({
                    "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    "ext": base_image["ext"],
                    "data": img_b64
                })

        document_data["pages"].append(page_dict)

    with open(output_json_path, 'w') as f:
        json.dump(document_data, f)
    
    print(f"Success. Exported cleaner layout data to {output_json_path}")

if __name__ == "__main__":
    parse_pdf_to_json("turtles.pdf", "assets/turtles_parsed.json")