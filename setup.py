import setuptools

setuptools.setup(
    name='makeitlearn-isga',
    version='2.5.1',
    author='Achraf Mataich, ISGA Fes',
    author_email='achraf.mataich@edu.isga.ma',
    description='This is a simple Machine Learning module',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    requires=["pandas", "numpy"],
    url="https://github.com/achrafmataich/ISGA_ML_Package"
)