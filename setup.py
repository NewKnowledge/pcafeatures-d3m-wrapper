from distutils.core import setup

setup(name='PcafeaturesD3MWrapper',
    version='3.0.1',
    description='A wrapper for running the punk pcafeatures functions in the d3m environment.',
    packages=['PcafeaturesD3MWrapper'],
    install_requires=["numpy==1.15.4",
        "pandas==0.23.4",
        "punk==3.0.0"],
    dependency_links=[
    ],
    entry_points = {
        'd3m.primitives': [
            'feature_selection.distil.Pcafeatures = PcafeaturesD3MWrapper:pcafeatures'
        ],
    },
)
