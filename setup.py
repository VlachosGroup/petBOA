from setuptools import setup, find_packages
	
setup(
    name="petboa",
    version="0.1.0",
    description="Parameter Estimation using Bayesian Optimization",
    author="Sashank Kasiraju; Vlachos Group",
    author_email="skasiraj@udel.edu",
    url="https://github.com/VlachosGroup/petboa",
    project_urls={
        "Documentation": "https://github.com/VlachosGroup/petboa",
        "Source": "https://github.com/VlachosGroup/petboa",
    },
    packages=find_packages(),
    python_requires=">=3.7,<=3.11",
    install_requires=[
        "gpytorch<=1.8.1",
        "botorch<=0.6.6",
        "nextorch>=0.1.0",
        "pmutt",
	"SALib"
    ],
    classifiers=[
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering :: Chemistry",
    ],
)
