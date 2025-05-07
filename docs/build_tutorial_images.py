import os
from inspect import cleandoc as multi_line_str

TUTORIAL_BUILD_PATH = "build_tutorial"


if not os.path.exists(TUTORIAL_BUILD_PATH):
    os.makedirs(TUTORIAL_BUILD_PATH)


def get_figures_names(file_name):
    """Getting the names of the figures created in the tutorial.
    
    Args:
        file_name (str): Path to the .rst tutorial file.

    Returns:
        list[str]: Names of the figures.
    """

    with open(file_name) as file:
        lines = file.readlines()

    figures_names = []

    for line in lines:
        if line.startswith(".. image::"):
            figures_names.append(line[10:].replace(" ", "").replace("\n", ""))
    
    return figures_names


def extract_code(file_name):
    """Extract the python code from the tutorial.
    
    Args:
        file_name (str): Path to the .rst tutorial file.

    Returns:
        str: Python code from the tutorial.
    """

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

    return python_code
    

def get_figures_saving_code(figures_names):
    """Build the code to save the figures created when running the tutorial code.
    
    Args:
        list[str]: Names of the figures.

    Returns:
        str: Python code.
    """

    python_code = ""

    python_code += f"TUTORIAL_BUILD_PATH = '{TUTORIAL_BUILD_PATH}'\n"
    python_code += multi_line_str("""
    # Saving figures
    
    import os

    figures_names = {figures_names}

    n_figures_created = len(plt.get_fignums())
    n_figures_found_in_doc = len(figures_names)
    assert n_figures_created == n_figures_found_in_doc

    for i in plt.get_fignums():
        fig_name = figures_names[i-1]
        dir = "source/" + os.path.dirname(fig_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.figure(i).savefig("source/" + fig_name)
    """.format(figures_names=figures_names))

    return python_code


tutorial_file_name = "source/tutorial.rst"
figures_names = get_figures_names(tutorial_file_name)
python_code = extract_code(tutorial_file_name)
python_code = "import torch; torch.set_num_threads(6)\n" + python_code
python_code = python_code + get_figures_saving_code(figures_names) + "\n"

output_file_name = f"{TUTORIAL_BUILD_PATH}/tutorial.py"
with open(output_file_name, "w+") as file:
    file.writelines(python_code)

os.system(f"python {output_file_name}")

