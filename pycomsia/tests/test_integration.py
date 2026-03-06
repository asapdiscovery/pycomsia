"""
Integration tests for PyCoMSIA full workflow.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from pycomsia.data_loader import DataLoader
from pycomsia.molecular_grid_calculator import MolecularGridCalculator
from pycomsia.molecular_field_calculator import MolecularFieldCalculator
from pycomsia.pls_analysis import PLSAnalysis
from pycomsia.contour_plot_visualizer import ContourPlotVisualizer


@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent.parent / "data"


class TestPyCoMSIAIntegration:
    """Integration tests for complete PyCoMSIA workflows."""
    
    def test_full_workflow_ace_dataset(self, test_data_dir):
        """Test complete workflow on ACE dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Load data
            data_loader = DataLoader()
            ace_file = test_data_dir / "ACE_train.sdf"
            smiles_list, mols, activities = data_loader.load_sdf_data(
                str(ace_file), "Activity", is_training=True
            )
            
            # Use subset for faster testing
            mols = mols[:20]
            activities = activities[:20]
            smiles_list = smiles_list[:20]
            
            # Validate data loading
            assert len(mols) == 20
            assert len(activities) == 20
            assert len(smiles_list) == 20
            
            # 2. Generate molecular grid
            grid_calc = MolecularGridCalculator()
            grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
                mols, resolution=2.5, padding=4.0
            )
            
            # Validate grid
            assert len(grid_dimensions) == 3
            assert all(dim > 0 for dim in grid_dimensions)
            
            # 3. Calculate molecular fields
            field_calc = MolecularFieldCalculator()
            all_fields = field_calc.calc_field(
                mols, grid_spacing, grid_dimensions, grid_origin
            )
            
            # Validate fields
            assert 'train_fields' in all_fields
            train_fields = all_fields['train_fields']
            assert 'steric_field' in train_fields
            assert 'electrostatic_field' in train_fields
            assert 'hydrophobic_field' in train_fields
            assert len(train_fields['steric_field']) == len(mols)
            
            # Convert to format expected by PLS analysis
            fields = {}
            for field_name, field_arrays in train_fields.items():
                if field_name in ['steric_field', 'electrostatic_field', 'hydrophobic_field']:
                    # Average fields across molecules for PLS
                    avg_field = np.mean([arr for arr in field_arrays], axis=0)
                    fields[field_name] = avg_field
            
            # 4. Perform PLS analysis
            pls_analysis = PLSAnalysis()
            pls_analysis.convert_fields_to_X(fields)
            pls_analysis.perform_loo_analysis(activities, max_components=5)
            pls_analysis.fit_final_model(activities, test_size=0.3)
            
            # Validate PLS results
            assert pls_analysis.pls_model is not None
            assert 1 <= pls_analysis.optimal_components <= 5
            assert 'r2' in pls_analysis.train_metrics
            assert pls_analysis.train_metrics['r2'] > 0
            
            # 5. Get coefficients for visualization
            coefficients = pls_analysis.get_coefficient_fields()
            
            # Validate coefficients
            assert 'steric_field' in coefficients
            assert 'electrostatic_field' in coefficients
            assert 'hydrophobic_field' in coefficients
            assert coefficients['steric_field'].shape == (expected_size,)
            
            # 6. Export PLS results
            pls_analysis.export_metrics_to_csv(temp_dir)
            pls_analysis.export_predictions_and_residuals(temp_dir)
            
            # Validate exports
            metrics_file = Path(temp_dir) / "PLS_Analysis" / "pls_model_metrics.csv"
            residuals_file = Path(temp_dir) / "PLS_Analysis" / "pls_residuals.csv"
            assert metrics_file.exists()
            assert residuals_file.exists()
            
            # 7. Generate contour plots (mocked for testing)
            contour_visualizer = ContourPlotVisualizer()
            significant_ranges = contour_visualizer.calculate_significant_ranges(coefficients)
            
            # Validate contour data
            assert 'steric_field' in significant_ranges
            assert 'electrostatic_field' in significant_ranges
            for field_name, ranges in significant_ranges.items():
                assert 'low' in ranges and 'high' in ranges
    
    def test_multiple_dataset_consistency(self, test_data_dir):
        """Test workflow consistency across different datasets."""
        datasets = ["ACE", "AChE", "CCR5"]
        results = {}
        
        for dataset in datasets:
            train_file = test_data_dir / f"{dataset}_train.sdf"
            if not train_file.exists():
                continue
                
            # Run abbreviated workflow
            data_loader = DataLoader()
            _, mols, activities = data_loader.load_sdf_data(
                str(train_file), "Activity", is_training=True
            )
            
            # Use small subset for speed
            mols = mols[:10]
            activities = activities[:10]
            
            # Generate grid
            grid_calc = MolecularGridCalculator()
            grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
                mols, resolution=3.0, padding=4.0
            )
            
            # Calculate steric field only for speed
            field_calc = MolecularFieldCalculator()
            all_fields = field_calc.calc_field(
                mols, grid_spacing, grid_dimensions, grid_origin
            )
            
            # Extract training fields and convert to expected format
            train_fields = all_fields['train_fields']
            fields = {}
            for field_name, field_arrays in train_fields.items():
                if field_name in ['steric_field']:
                    # Average fields across molecules for PLS
                    avg_field = np.mean([arr for arr in field_arrays], axis=0)
                    fields[field_name] = avg_field
            
            # PLS analysis
            pls_analysis = PLSAnalysis()
            pls_analysis.convert_fields_to_X(fields)
            pls_analysis.perform_loo_analysis(activities, max_components=3)
            pls_analysis.fit_final_model(activities)
            
            results[dataset] = {
                'molecules': len(mols),
                'grid_size': np.prod(grid_dimensions),
                'r2_train': pls_analysis.train_metrics['r2'],
                'optimal_components': pls_analysis.optimal_components
            }
        
        # Validate that all datasets produced reasonable results
        for dataset, result in results.items():
            assert result['molecules'] == 10
            assert result['grid_size'] > 0
            assert result['r2_train'] > 0  # Should have some predictive power
            assert 1 <= result['optimal_components'] <= 3
    
    def test_train_test_split_workflow(self, test_data_dir):
        """Test workflow with train/test split using separate files."""
        ace_train_file = test_data_dir / "ACE_train.sdf"
        ace_test_file = test_data_dir / "ACE_test.sdf"
        
        if not (ace_train_file.exists() and ace_test_file.exists()):
            pytest.skip("ACE train/test files not available")
        
        # 1. Load training data
        data_loader = DataLoader()
        _, train_mols, train_activities = data_loader.load_sdf_data(
            str(ace_train_file), "Activity", is_training=True
        )
        
        # 2. Load test data
        _, test_mols, _ = data_loader.load_sdf_data(
            str(ace_test_file), is_training=False
        )
        
        # Use subsets for speed
        train_mols = train_mols[:15]
        train_activities = train_activities[:15]
        test_mols = test_mols[:5]
        
        # 3. Combine molecules for grid generation
        all_mols = train_mols + test_mols
        
        # 4. Generate grid
        grid_calc = MolecularGridCalculator()
        grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
            all_mols, resolution=2.5, padding=4.0
        )
        
        # 5. Calculate fields for training molecules
        field_calc = MolecularFieldCalculator()
        all_train_fields = field_calc.calc_field(
            train_mols, grid_spacing, grid_dimensions, grid_origin
        )
        
        # 6. Calculate fields for test molecules
        all_test_fields = field_calc.calc_field(
            test_mols, grid_spacing, grid_dimensions, grid_origin
        )
        
        # Convert to expected format
        train_fields = {}
        test_fields = {}
        
        for field_name, field_arrays in all_train_fields['train_fields'].items():
            if field_name in ['steric_field', 'electrostatic_field']:
                avg_field = np.mean([arr for arr in field_arrays], axis=0)
                train_fields[field_name] = avg_field
                
        for field_name, field_arrays in all_test_fields['train_fields'].items():
            if field_name in ['steric_field', 'electrostatic_field']:
                avg_field = np.mean([arr for arr in field_arrays], axis=0)
                test_fields[field_name] = avg_field
        
        # 7. PLS analysis with separate training and prediction
        pls_analysis = PLSAnalysis()
        pls_analysis.convert_fields_to_X(train_fields, test_fields)
        pls_analysis.perform_loo_analysis(train_activities, max_components=3)
        pls_analysis.fit_final_model(train_activities, test_size=0.0)  # Use all training data
        
        # 8. Validate that we can make predictions
        assert pls_analysis.X_pred is not None  # Test features should be available
        assert pls_analysis.X_pred.shape[0] == len(test_mols)
        
        # 9. Get coefficients
        coefficients = pls_analysis.get_coefficient_fields()
        assert 'steric_field' in coefficients
        assert 'electrostatic_field' in coefficients
    
    def test_edge_cases_and_error_handling(self, test_data_dir):
        """Test edge cases and error handling in the workflow."""
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        
        # Test with single molecule
        _, mols, activities = data_loader.load_sdf_data(
            str(ace_file), "Activity", is_training=True
        )
        single_mol = mols[:1]
        single_activity = activities[:1]
        
        # Grid calculation should work with single molecule
        grid_calc = MolecularGridCalculator()
        grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
            single_mol, resolution=3.0, padding=4.0
        )
        assert all(dim > 0 for dim in grid_dimensions)
        
        # Field calculation should work with single molecule
        field_calc = MolecularFieldCalculator()
        all_fields = field_calc.calc_field(
            single_mol, grid_spacing, grid_dimensions, grid_origin
        )
        
        train_fields = all_fields['train_fields']
        assert 'steric_field' in train_fields
        assert len(train_fields['steric_field']) == 1  # One molecule
        
        # PLS with single molecule should handle gracefully
        pls_analysis = PLSAnalysis()
        pls_analysis.convert_fields_to_X(fields)
        
        # Should handle small dataset appropriately
        with pytest.raises(Exception):  # PLS needs multiple molecules
            pls_analysis.perform_loo_analysis(single_activity, max_components=5)


class TestWorkflowRegression:
    """Regression tests for complete workflow to prevent performance drift."""
    
    def test_ace_workflow_performance_benchmark(self, test_data_dir):
        """Benchmark test for ACE workflow performance."""
        # This test establishes performance baselines
        
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, activities = data_loader.load_sdf_data(
            str(ace_file), "Activity", is_training=True
        )
        
        # Use standard parameters
        grid_calc = MolecularGridCalculator()
        grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
            mols, resolution=2.0, padding=4.0
        )
        
        field_calc = MolecularFieldCalculator()
        all_fields = field_calc.calc_field(
            mols, grid_spacing, grid_dimensions, grid_origin
        )
        
        # Extract training fields
        fields = all_fields['train_fields']
        
        pls_analysis = PLSAnalysis()
        pls_analysis.convert_fields_to_X(fields)
        pls_analysis.perform_loo_analysis(activities, max_components=10)
        pls_analysis.fit_final_model(activities, test_size=0.2, random_state=42)
        
        # Performance benchmarks (these should be stable)
        train_r2 = pls_analysis.train_metrics['r2']
        test_r2 = pls_analysis.test_metrics['r2']
        optimal_components = pls_analysis.optimal_components
        
        # Expected performance ranges for ACE dataset
        assert 0.7 <= train_r2 <= 0.95, f"Training R² regression: {train_r2}"
        assert 0.4 <= test_r2 <= 0.8, f"Test R² regression: {test_r2}"
        assert 2 <= optimal_components <= 6, f"Optimal components regression: {optimal_components}"
        
        # Coefficient quality checks
        coefficients = pls_analysis.get_coefficient_fields()
        steric_coeff_std = np.std(coefficients['steric_field'])
        electro_coeff_std = np.std(coefficients['electrostatic_field'])
        
        assert 0.1 <= steric_coeff_std <= 2.0, f"Steric coefficient std regression: {steric_coeff_std}"
        assert 0.1 <= electro_coeff_std <= 2.0, f"Electro coefficient std regression: {electro_coeff_std}"
    
    def test_field_contribution_stability(self, test_data_dir):
        """Test that field contributions are stable across runs."""
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, activities = data_loader.load_sdf_data(
            str(ace_file), "Activity", is_training=True
        )
        mols = mols[:30]  # Subset for speed
        activities = activities[:30]
        
        # Run workflow twice with same random seed
        contribution_results = []
        
        for run in range(2):
            grid_calc = MolecularGridCalculator()
            grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
                mols, resolution=2.5, padding=4.0
            )
            
            field_calc = MolecularFieldCalculator()
            all_fields = field_calc.calc_field(
                mols, grid_spacing, grid_dimensions, grid_origin
            )
            
            # Extract training fields
            fields = all_fields['train_fields']
            
            pls_analysis = PLSAnalysis()
            pls_analysis.convert_fields_to_X(fields)
            pls_analysis.perform_loo_analysis(activities, max_components=5)
            pls_analysis.fit_final_model(activities, random_state=42)  # Fixed seed
            
            fractions = pls_analysis.get_field_contribution_fractions()
            contribution_results.append(fractions)
        
        # Results should be very similar across runs
        for field_name in contribution_results[0].keys():
            frac1 = contribution_results[0][field_name]
            frac2 = contribution_results[1][field_name]
            assert abs(frac1 - frac2) < 0.05, \
                f"{field_name} contribution unstable: {frac1} vs {frac2}"
    
    @patch('pycomsia.contour_plot_visualizer.PYMOL_AVAILABLE', True)
    def test_visualization_integration_structure(self, test_data_dir):
        """Test that visualization integration maintains expected structure."""
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, activities = data_loader.load_sdf_data(
            str(ace_file), "Activity", is_training=True
        )
        
        # Small subset for testing
        mols = mols[:5]
        activities = activities[:5]
        
        # Complete workflow
        grid_calc = MolecularGridCalculator()
        grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
            mols, resolution=3.0, padding=4.0
        )
        
        field_calc = MolecularFieldCalculator()
        all_fields = field_calc.calc_field(
            mols, grid_spacing, grid_dimensions, grid_origin
        )
        
        # Extract training fields
        fields = all_fields['train_fields']
        
        pls_analysis = PLSAnalysis()
        pls_analysis.convert_fields_to_X(fields)
        pls_analysis.perform_loo_analysis(activities, max_components=3)
        pls_analysis.fit_final_model(activities)
        coefficients = pls_analysis.get_coefficient_fields()
        
        # Test contour visualization setup
        contour_visualizer = ContourPlotVisualizer()
        significant_ranges = contour_visualizer.calculate_significant_ranges(coefficients)
        
        # Validate visualization data structure
        assert 'steric_field' in significant_ranges
        assert 'electrostatic_field' in significant_ranges
        
        for field_name, ranges in significant_ranges.items():
            # Each field should have proper contour ranges
            assert 'low' in ranges and 'high' in ranges
            assert len(ranges['low']) == 2
            assert len(ranges['high']) == 2
            assert ranges['low'][0] < ranges['low'][1] <= 0
            assert 0 <= ranges['high'][0] < ranges['high'][1]
        
        # Test that molecules are properly formatted for visualization
        molecules_for_viz = [mol for mol, _ in mols]
        assert len(molecules_for_viz) == 5
        assert all(mol is not None for mol in molecules_for_viz)
        assert all(mol.GetNumConformers() > 0 for mol in molecules_for_viz)