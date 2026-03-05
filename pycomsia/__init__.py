"""PyCoMSIA: Comparative Molecular Similarity Indices Analysis."""

from ._version import __version__

# Lazy imports to avoid heavy dependencies during CLI access
def __getattr__(name):
    """Lazy import of modules to avoid loading heavy dependencies unless needed."""
    if name == "DataLoader":
        from .data_loader import DataLoader
        return DataLoader
    elif name == "MoleculeAligner":
        from .molecule_aligner import MoleculeAligner
        return MoleculeAligner
    elif name == "MolecularGridCalculator":
        from .molecular_grid_calculator import MolecularGridCalculator
        return MolecularGridCalculator
    elif name == "MolecularFieldCalculator":
        from .molecular_field_calculator import MolecularFieldCalculator
        return MolecularFieldCalculator
    elif name == "MolecularVisualizer":
        from .molecular_visualizer import MolecularVisualizer
        return MolecularVisualizer
    elif name == "PLSAnalysis":
        from .pls_analysis import PLSAnalysis
        return PLSAnalysis
    elif name == "PLSAnalysisTestSets":
        from .pls_analysis_test_sets import PLSAnalysisTestSets
        return PLSAnalysisTestSets
    elif name == "ContourPlotVisualizer":
        from .contour_plot_visualizer import ContourPlotVisualizer
        return ContourPlotVisualizer
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Export all classes
__all__ = [
    'DataLoader',
    'MoleculeAligner', 
    'MolecularGridCalculator',
    'MolecularFieldCalculator',
    'MolecularVisualizer', 
    'PLSAnalysis',
    'PLSAnalysisTestSets',
    'ContourPlotVisualizer'
]
