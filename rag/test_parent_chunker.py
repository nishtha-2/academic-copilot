import os

# 1. Corrected the import to match your pdf_loader.py function
from pdf_loader import extract_pages

from parent_chunker import (
    create_parent_chunks,
    create_child_chunks
)

DOCUMENT_FOLDER = "../data/documents"

# -----------------------------
# 2. List available PDFs
# -----------------------------
pdfs = sorted(
    [
        file
        for file in os.listdir(DOCUMENT_FOLDER)
        if file.endswith(".pdf")
    ]
)

print("\nAvailable PDFs\n")
for i, pdf in enumerate(pdfs, start=1):
    print(f"{i}. {pdf}")

# -----------------------------
# 3. User Selection
# -----------------------------
choice = int(
    input("\nSelect PDF Number: ")
)
selected_pdf = pdfs[choice - 1]
pdf_path = os.path.join(
    DOCUMENT_FOLDER,
    selected_pdf
)

print("\nLoading PDF...")
# Extract the list of page dictionaries
pages_data = extract_pages(pdf_path)

# Flatten the text across all pages into a single string for the chunker
text = "\n".join([page["text"] for page in pages_data])

# -----------------------------
# 4. Chunking Process
# -----------------------------
parents = create_parent_chunks(
    text,
    selected_pdf
)

children = create_child_chunks(
    parents
)

# -----------------------------
# 5. Summary Results
# -----------------------------
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\nSelected PDF : {selected_pdf}")
print(f"Parent Chunks : {len(parents)}")
print(f"Child Chunks  : {len(children)}")

# -----------------------------
# 6. Sample Output Verification
# -----------------------------
print("\n" + "=" * 70)
print("FIRST PARENT CHUNK")
print("=" * 70)
if parents:
    print(parents[0]["text"][:1200])
else:
    print("No parent chunks generated.")

print("\n" + "=" * 70)
print("FIRST CHILD CHUNK")
print("=" * 70)
if children:
    print(children[0]["text"])
else:
    print("No child chunks generated.")

print("\n" + "=" * 70)
print("CHILD METADATA")
print("=" * 70)
if children:
    print(children[0])

# -----------------------------
# 7. Parent-Child Relationship Mapping Lookup
# -----------------------------
print("\n" + "=" * 70)
print("PARENT OF FIRST CHILD")
print("=" * 70)

if children and parents:
    parent_id = children[0]["parent_id"]
    
    # Safe lookup logic: handles both integer index IDs and unique string keys/UUIDs
    if isinstance(parent_id, int) and parent_id < len(parents):
        print(parents[parent_id]["text"][:1200])
    else:
        # Fallback if your chunker assigns string/UUID identifiers instead of list indexes
        parent_chunk = next((p for p in parents if p.get("id") == parent_id or p.get("parent_id") == parent_id), None)
        if parent_chunk:
            print(parent_chunk["text"][:1200])
        else:
            print(f"Could not find matching parent chunk for ID: {parent_id}")

# -----------------------------
# 8. Validation Check
# -----------------------------
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

valid = True
parent_ids_pool = {p.get("id") for p in parents if "id" in p}

for child in children:
    pid = child["parent_id"]
    
    if isinstance(pid, int):
        if pid >= len(parents):
            valid = False
            print(f"Invalid integer parent_id index: {pid}")
    else:
        if pid not in parent_ids_pool:
            valid = False
            print(f"Invalid string parent_id reference: {pid}")

if valid and children:
    print("All child chunks correctly reference a parent.")
elif not children:
    print("No chunks to validate.")