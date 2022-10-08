import setuptools

setuptools.setup(
    name='makeitlearn-isga',
    version='2.5.2',
    author='Achraf Mataich, ISGA Fes',
    author_email='achraf.mataich@edu.isga.ma',
    description='This is a simple Machine Learning module',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
    ],
)