from distutils.core import setup

setup(name='PcafeaturesD3MWrapper',
    version='3.0.0',
    description='A wrapper for running the punk pcafeatures functions in the d3m environment.',
    packages=['PcafeaturesD3MWrapper'],
    install_requires=["numpy",
        "pandas",
        "requests",
        "typing",
        "punk==3.0.0"],
    dependency_links=[
    ],
    entry_points = {
        'd3m.primitives': [
            'distil.pcafeatures = PcafeaturesD3MWrapper:pcafeatures'
        ],
    },
)
