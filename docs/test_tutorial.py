import os
from inspect import cleandoc as multi_line_str

TUTORIAL_BUILD_PATH = "build_tutorial"


if not os.path.exists(TUTORIAL_BUILD_PATH):
    os.makedirs(TUTORIAL_BUILD_PATH)


def extract_code(file_name):
    with open(file_name) as file:
        lines = file.readlines()

    python_code = ""

    in_code = False
    code_start = False
    for line in lines:
        if line.startswith(".. code-block:: python"):
            in_code = True
            code_start = True
        elif not line.startswith("    ") and len(line.replace(" ", "")) > 1:
            in_code = False
        else:
            code_start = False
        
        if in_code and not code_start:
            code_line = line[4:]
            if len(code_line) == 0:
                python_code += "\n"
            else:
                python_code += line[4:]

    # Adding by hand the saving of the figures
    python_code += f"TUTORIAL_BUILD_PATH = '{TUTORIAL_BUILD_PATH}'\n"
    python_code += multi_line_str("""
    # Saving figures
    
    import os

    dir = f"{TUTORIAL_BUILD_PATH}/figures"
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in plt.get_fignums():
        plt.figure(i).savefig(f"{dir}/{i}.png")
    """)

    return python_code
    

tutorial_file_name = "source/tutorial.rst"
python_code = extract_code(tutorial_file_name)
python_code = "import torch; torch.set_num_threads(6)\n" + python_code

output_file_name = f"{TUTORIAL_BUILD_PATH}/tutorial.py"
with open(output_file_name, "w+") as file:
    file.writelines(python_code)

os.system(f"python {output_file_name}")

