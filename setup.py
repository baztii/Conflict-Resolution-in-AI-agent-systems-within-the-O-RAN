from setuptools import setup, find_packages

setup(
    name='Conflict-Resolution-in-AI-agent-systems-within-the-O-RAN',
    version='0.0.1',
    description='AI agents in the O-RAN environment',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # Indica el formato del README
    author='Miquel P. Baztan Grau',
    author_email='miquelbaztan@gmail.com',
    github='',
    gitlab='',
    packages=find_packages(),  # Encuentra y lista todos los paquetes del proyecto
    install_requires=[
        'dependencia1>=1.0',
        'dependencia2>=2.0',
    ],  # Lista de dependencias que se instalarán con el paquete
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: CC0 1.0 Universal',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
    license='CC0 1.0 Universal'
)
