import os

import mkdocs_gen_files


def process_file(filename):
    with open(filename) as f:
        content = f.read()
    # headline_index = 0
    # for line in content:
    #     if line.startswith("# "):
    #         break
    #     headline_index += 1
    # content = content[headline_index:]

    filename = os.path.dirname(filename.removeprefix("multimolecule/")) + ".md"

    with mkdocs_gen_files.open(filename, "w") as file:
        print(content, file=file)

    mkdocs_gen_files.set_edit_path(filename, "gen_pages.py")


for root, _dirs, files in os.walk("multimolecule/models"):
    if "README.md" in files:
        process_file(os.path.join(root, "README.md"))
