"""
Unit tests for PLSAnalysis module.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from pycomsia.data_loader import DataLoader
from pycomsia.molecular_grid_calculator import MolecularGridCalculator
from pycomsia.molecular_field_calculator import MolecularFieldCalculator
from pycomsia.pls_analysis import PLSAnalysis


@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def pls_analysis():
    """Create PLSAnalysis instance."""
    return PLSAnalysis()


@pytest.fixture
def test_fields_and_activities(test_data_dir):
    """Generate test fields and activities for PLS testing."""
    # Load ACE molecules
    data_loader = DataLoader()
    ace_file = test_data_dir / "ACE_train.sdf"
    _, mols, activities = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
    
    # Use very small subset for faster testing
    mols = mols[:8]
    activities = activities[:8]
    
    # Generate grid
    grid_calc = MolecularGridCalculator()
    grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
        mols, resolution=2.5, padding=3.0
    )
    
    # Calculate fields
    field_calc = MolecularFieldCalculator()
    all_fields = field_calc.calc_field(
        mols, grid_spacing, grid_dimensions, grid_origin
    )
    
    # Use the training fields directly - this matches the actual API
    train_fields = all_fields['train_fields']
    
    return train_fields, activities, grid_dimensions, grid_origin, grid_spacing


class TestPLSAnalysis:
    """Test PLSAnalysis functionality."""
    
    def test_init(self, pls_analysis):
        """Test PLSAnalysis initialization."""
        assert pls_analysis.pls_model is None
        assert pls_analysis.X_train is None
        assert pls_analysis.y is None
        assert pls_analysis.scalers is None
    
    def test_convert_fields_to_X(self, pls_analysis, test_fields_and_activities):
        """Test conversion of field data to feature matrix."""
        fields, activities, grid_dimensions, _, _ = test_fields_and_activities
        
        pls_analysis.convert_fields_to_X(fields)
        
        # Check that feature matrix was created
        assert pls_analysis.X_train is not None
        assert pls_analysis.X_train.shape[0] == len(activities)  # Number of molecules
        
        # Number of features should be grid size * number of fields
        expected_features = np.prod(grid_dimensions) * len(fields)
        assert pls_analysis.X_train.shape[1] == expected_features
        
        # Should contain finite values
        assert np.all(np.isfinite(pls_analysis.X_train))
    
    def test_perform_loo_analysis(self, pls_analysis, test_fields_and_activities):
        """Test leave-one-out cross-validation analysis."""
        fields, activities, _, _, _ = test_fields_and_activities
        
        # Convert fields to feature matrix
        pls_analysis.convert_fields_to_X(fields)
        
        # Perform LOO analysis
        pls_analysis.perform_loo_analysis(activities, max_components=5)
        
        # Check that analysis results are stored
        assert hasattr(pls_analysis, 'q2_scores')
        assert len(pls_analysis.q2_scores) == 5
        assert pls_analysis.optimal_n_components > 0
        assert pls_analysis.optimal_n_components <= 5
        
        # Q2 scores should be reasonable
        q2_scores = pls_analysis.q2_scores
        assert all(-1 <= q2 <= 1 for q2 in q2_scores)
    
    def test_fit_final_model(self, pls_analysis, test_fields_and_activities):
        """Test fitting final PLS model."""
        fields, activities, _, _, _ = test_fields_and_activities
        
        # Prepare data
        pls_analysis.convert_fields_to_X(fields)
        pls_analysis.perform_loo_analysis(activities, max_components=3)
        
        # Fit final model
        pls_analysis.fit_final_model(activities, test_size=0.3)
        
        # Check that model was fitted
        assert pls_analysis.pls_model is not None
        assert pls_analysis.y_train_final is not None
        assert pls_analysis.X_train_final is not None
        
        # Check metrics (these get calculated in fit_final_model)
        assert pls_analysis.r2_train is not None
        assert pls_analysis.r2_test is not None
    
    def test_get_coefficient_fields(self, pls_analysis, test_fields_and_activities):
        """Test extraction of coefficient fields."""
        fields, activities, grid_dimensions, _, _ = test_fields_and_activities
        
        # Prepare and fit model
        pls_analysis.convert_fields_to_X(fields)
        pls_analysis.perform_loo_analysis(activities, max_components=3)
        pls_analysis.fit_final_model(activities)
        
        # Get coefficients
        coefficients = pls_analysis.get_coefficient_fields()
        
        # Check structure
        assert isinstance(coefficients, dict)
        assert 'steric_field' in coefficients
        assert 'electrostatic_field' in coefficients
        
        # Check shapes
        expected_size = np.prod(grid_dimensions)
        assert coefficients['steric_field'].shape == (expected_size,)
        assert coefficients['electrostatic_field'].shape == (expected_size,)
        
        # Coefficients should be finite
        assert np.all(np.isfinite(coefficients['steric_field']))
        assert np.all(np.isfinite(coefficients['electrostatic_field']))


class TestPLSAnalysisRegression:
    """Regression tests for PLS analysis to prevent drift."""
    
    def test_ace_pls_performance(self, test_data_dir):
        """Test PLS performance on full ACE dataset (regression test)."""
        # Load full ACE dataset
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, activities = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
        
        # Generate grid and fields
        grid_calc = MolecularGridCalculator()
        grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
            mols, resolution=2.0, padding=4.0
        )
        
        field_calc = MolecularFieldCalculator()
        all_fields = field_calc.calc_field(
            mols, grid_spacing, grid_dimensions, grid_origin
        )
        
        # Convert to expected format for PLS
        train_fields = all_fields['train_fields']
        fields = {}
        for field_name, field_arrays in train_fields.items():
            if field_name in ['steric_field', 'electrostatic_field']:  # Only use these for testing
                # Average the fields across molecules
                avg_field = np.mean([arr for arr in field_arrays], axis=0)
                fields[field_name] = avg_field
        
        # Perform PLS analysis
        pls_analysis = PLSAnalysis()
        pls_analysis.convert_fields_to_X(fields)
        pls_analysis.perform_loo_analysis(activities, max_components=10)
        pls_analysis.fit_final_model(activities, test_size=0.2)
        
        # Performance should be reasonable for ACE dataset
        train_r2 = pls_analysis.r2_train
        test_r2 = pls_analysis.r2_test
        
        # Expected performance thresholds
        assert train_r2 >= 0.6, f"Training R² too low: {train_r2}"
        assert test_r2 >= 0.3, f"Test R² too low: {test_r2}"
        assert train_r2 >= test_r2, "Training R² should be >= test R²"
        
        # Optimal components should be reasonable
        assert 1 <= pls_analysis.optimal_n_components <= 8, \
            f"Optimal components out of range: {pls_analysis.optimal_n_components}"
    
    def test_coefficient_magnitude_stability(self, pls_analysis, test_fields_and_activities):
        """Test that coefficient magnitudes are stable (regression test)."""
        fields, activities, _, _, _ = test_fields_and_activities
        
        # Fit model
        pls_analysis.convert_fields_to_X(fields)
        pls_analysis.perform_loo_analysis(activities, max_components=3)
        pls_analysis.fit_final_model(activities)
        
        # Get coefficients
        coefficients = pls_analysis.get_coefficient_fields()
        
        # Check coefficient magnitudes
        for field_name, coeff_values in coefficients.items():
            coeff_std = np.std(coeff_values)
            coeff_max = np.max(np.abs(coeff_values))
            
            # Coefficients should have reasonable magnitude
            assert 0.001 <= coeff_std <= 10.0, \
                f"{field_name} coefficient std out of range: {coeff_std}"
            assert 0.001 <= coeff_max <= 50.0, \
                f"{field_name} coefficient max out of range: {coeff_max}"
    
    def test_field_contribution_fractions(self, test_data_dir):
        """Test field contribution fractions calculation."""
        # Load small dataset for testing
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, activities = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
        mols = mols[:30]  # Subset for speed
        activities = activities[:30]
        
        # Generate fields
        grid_calc = MolecularGridCalculator()
        grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
            mols, resolution=2.5, padding=4.0
        )
        
        field_calc = MolecularFieldCalculator()
        all_fields = field_calc.calc_field(
            mols, grid_spacing, grid_dimensions, grid_origin
        )
        
        # Convert to expected format for PLS
        train_fields = all_fields['train_fields']
        fields = {}
        for field_name, field_arrays in train_fields.items():
            if field_name in ['steric_field', 'electrostatic_field', 'hydrophobic_field']:  
                # Average the fields across molecules
                avg_field = np.mean([arr for arr in field_arrays], axis=0)
                fields[field_name] = avg_field
        
        # Perform PLS analysis
        pls_analysis = PLSAnalysis()
        pls_analysis.convert_fields_to_X(fields)
        pls_analysis.perform_loo_analysis(activities, max_components=5)
        pls_analysis.fit_final_model(activities)
        
        # Get contribution fractions
        fractions = pls_analysis.get_field_contribution_fractions()
        
        # Check that fractions sum to 1
        total_fraction = sum(fractions.values())
        assert abs(total_fraction - 1.0) < 0.01, f"Fractions should sum to 1: {total_fraction}"
        
        # All fractions should be positive
        for field_name, fraction in fractions.items():
            assert fraction >= 0, f"{field_name} fraction should be non-negative: {fraction}"
            assert fraction <= 1, f"{field_name} fraction should be <= 1: {fraction}"
    
    def test_export_functionality(self, pls_analysis, test_fields_and_activities):
        """Test that export methods work without errors."""
        fields, activities, _, _, _ = test_fields_and_activities
        
        # Fit model
        pls_analysis.convert_fields_to_X(fields)
        pls_analysis.perform_loo_analysis(activities, max_components=3)
        pls_analysis.fit_final_model(activities)
        
        # Test export methods with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # These should not raise errors
            pls_analysis.export_metrics_to_csv(temp_dir)
            pls_analysis.export_predictions_and_residuals(temp_dir)
            
            # Check that files were created
            metrics_file = Path(temp_dir) / "PLS_Analysis" / "pls_model_metrics.csv"
            residuals_file = Path(temp_dir) / "PLS_Analysis" / "pls_residuals.csv"
            
            assert metrics_file.exists(), "Metrics file should be created"
            assert residuals_file.exists(), "Residuals file should be created"
    
    def test_deterministic_behavior_with_fixed_seed(self, pls_analysis, test_fields_and_activities):
        """Test that PLS analysis is deterministic with fixed random seed."""
        fields, activities, _, _, _ = test_fields_and_activities
        
        # First run with fixed seed
        np.random.seed(42)
        pls_analysis1 = PLSAnalysis()
        pls_analysis1.convert_fields_to_X(fields)
        pls_analysis1.perform_loo_analysis(activities, max_components=3)
        pls_analysis1.fit_final_model(activities, test_size=0.3, random_state=42)
        coeffs1 = pls_analysis1.get_coefficient_fields()
        
        # Second run with same seed
        np.random.seed(42)
        pls_analysis2 = PLSAnalysis()
        pls_analysis2.convert_fields_to_X(fields)
        pls_analysis2.perform_loo_analysis(activities, max_components=3)
        pls_analysis2.fit_final_model(activities, test_size=0.3, random_state=42)
        coeffs2 = pls_analysis2.get_coefficient_fields()
        
        # Results should be very similar (allowing for minor numerical differences)
        for field_name in coeffs1.keys():
            np.testing.assert_allclose(
                coeffs1[field_name], coeffs2[field_name], 
                rtol=1e-10, atol=1e-10,
                err_msg=f"{field_name} coefficients should be deterministic"
            )