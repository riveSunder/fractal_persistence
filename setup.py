from setuptools import setup

setup(name="fracatal",\
        packages = ["fracatal", "tests"],\
        version = "0.0",\
        description = "Fractal boundaries for the persistence of motile pseudorganisms", \
        install_requires = ["numpy>=1.24.2",\
                "matplotlib==3.10.5",\
                "ipython==9.4.0",\
                "jax==0.7.0",\
                "jaxlib==0.7.0",\
                "mpi4py==4.1.0",\
                "scikit-image==0.25.2",\
                "torch==2.7.1"]\
        )

    
