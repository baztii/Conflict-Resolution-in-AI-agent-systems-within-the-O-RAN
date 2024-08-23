from setuptools import setup, find_packages

setup(
    name='Conflict-Resolution-in-AI-agent-systems-within-the-O-RAN',
    version='0.0.1',
    description='AI agents try learn the best policy for the O-RAN environment',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Miquel P. Baztan Grau',
    author_email='miquelbaztan@gmail.com',
    url='https://github.com/baztii/Conflict-Resolution-in-AI-agent-systems-within-the-O-RAN',
    packages=find_packages(),
    install_requires=[
        "torch",                 
        "pyyaml",                 
        "gymnasium",           
        "pyomo",                 
        "matplotlib",             
        "numpy",
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: CC0 1.0 Universal',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
    license='CC0 1.0 Universal',
    entry_points={
        'console_scripts': [
            'post_install_message = setup:post_install_message',
        ],
    }
)

def post_install_message():
    print(
        "\n"
        "---------------------------------------------------------\n"
        "¡Instalación completa!\n"
        "Por favor, asegúrate de que tienes instalados los solvers:\n"
        "  - SCIP: https://www.scipopt.org/\n"
        "  - IPOPT: https://coin-or.github.io/Ipopt/\n"
        "Agrega los solvers al PATH del sistema para que Pyomo pueda utilizarlos.\n"
        "---------------------------------------------------------\n"
    )