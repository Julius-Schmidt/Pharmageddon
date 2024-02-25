from setuptools import setup, find_packages 

setup( 
        name ='sepia', 
        version ='0.0.1', 
        author ='David Wagemann et al.', 
        author_email ='sepia@tum.de', 
        url ='https://github.com/Bioinformatics-Munich-Student-Lab/sepia', 
        description ='Package for predicting polypharmacy effects.', 
        long_description = "Tool which is able to predict the likelihood of drug-effect combinations for any number of interacting drugs.", 
        long_description_content_type ="text/markdown", 
        license ='TODO', 
        packages = find_packages(), 
        entry_points ={ 
            'console_scripts': [ 
                'sepia = src.main:main'
            ] 
        }, 
        include_package_data = True,
        classifiers =( 
            "Programming Language :: Python :: 3", 
            "License :: TODO", 
            "Operating System :: OS Independent", 
        ), 
        keywords ='polypharmacy drugs drug-interactions drug-effects medication bioinformatics', 
        zip_safe = False
) 