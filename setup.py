from setuptools import setup

setup(name="fracatal",\
        packages = ["fracatal", "tests"],\
        version = "0.0",\
        description = "Fractal boundaries for the persistence of motile pseudorganisms", \
        install_requires = ["numpy==1.24.2",\
                "matplotlib==3.7.0"]\
        )

    
