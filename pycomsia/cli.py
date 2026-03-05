"""Command-line interface for PyCoMSIA."""

import click
from datetime import datetime
import os
import numpy as np

# Import PyCoMSIA modules
from .data_loader import DataLoader
from .molecular_grid_calculator import MolecularGridCalculator
from .molecular_field_calculator import MolecularFieldCalculator

# Try to import visualizers, but handle missing dependencies gracefully
try:
    from .molecular_visualizer import MolecularVisualizer
    from .contour_plot_visualizer import ContourPlotVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Field options matching the original CLI
FIELD_OPTIONS = {
    "all": ["steric", "electrostatic", "hydrophobic", "hbond_donor", "hbond_acceptor"],
    "SE": ["steric", "electrostatic"],
    "SED": ["steric", "electrostatic", "hbond_donor"],
    "SEA": ["steric", "electrostatic", "hbond_acceptor"],
    "SEAD": ["steric", "electrostatic", "hbond_acceptor", "hbond_donor"],
    "SEDA": ["steric", "electrostatic", "hbond_acceptor", "hbond_donor"],
    "SEH": ["steric", "electrostatic", "hydrophobic"]
}


@click.command()
@click.option(
    '--train-file', '-t',
    required=True,
    type=click.Path(exists=True), 
    help='SDF file containing molecular structures and activities. Molecules MUST be prealigned.'
)
@click.option(
    '--predict-file', '-p',
    type=click.Path(exists=True),
    help='Path to the input SDF file for prediction. Molecules MUST be prealigned.'
)
@click.option(
    '--sdf-activity', '-a',
    required=True,
    type=str,
    help='Activity column name to use from the SDF file.'
)
@click.option(
    '--grid-resolution', '-r',
    default=1.0,
    type=float,
    help='Resolution of the grid used for field calculation. (default: 1.0)'
)
@click.option(
    '--grid-padding', '-g',
    default=3.0,
    type=float,
    help='Padding of the grid used for field calculation. (default: 3.0)'
)
@click.option(
    '--fields', '-f',
    default='all',
    type=click.Choice(list(FIELD_OPTIONS.keys())),
    help=f'Fields to use for analysis. Options: {", ".join(FIELD_OPTIONS.keys())} (default: all)'
)
@click.option(
    '--num-components', '-n',
    default=12,
    type=int,
    help='Number of components for PLS analysis. (default: 12)'
)
@click.option(
    '--column-filter', '-c',
    default=0.0,
    type=float,
    help='Column filtering. (default: 0.0)'
)
@click.option(
    '--disable-visualization', '-d',
    is_flag=True,
    help='Disable visualization (default: False)'
)
@click.option(
    '--output-dir', '-o',
    required=True,
    type=click.Path(),
    help='Output directory for results.'
)
@click.version_option()
def main(train_file, predict_file, sdf_activity, grid_resolution, grid_padding, 
         fields, num_components, column_filter, disable_visualization, output_dir):
    """Run PyCoMSIA molecular field analysis.
    
    PyCoMSIA performs Comparative Molecular Similarity Indices Analysis (CoMSIA)
    for drug discovery and molecular modeling applications.
    
    The analysis involves:
    1. Loading prealigned molecular data from SDF files
    2. Calculating molecular interaction fields (steric, electrostatic, etc.)
    3. Building PLS models to correlate fields with biological activity
    4. Generating visualizations and contour plots
    
    Examples:
    
    Basic analysis with SDF file:
    
        pycomsia -t molecules.sdf -a pIC50 -o results
    
    Analysis with prediction set:
    
        pycomsia -t train.sdf -p test.sdf -a pIC50 -o results
    
    Custom field selection and grid settings:
    
        pycomsia -t data.sdf -a Activity -f SE -r 0.5 -o results
    """
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        subdirectories = ["Contour_Plots", "PLS_Analysis", "Alignments", "Field_Plots"]
        for subdir in subdirectories:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    click.echo(f"🧬 PyCoMSIA Analysis Starting...")
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo(f"📄 Training file: {train_file}")
    if predict_file:
        click.echo(f"🔮 Prediction file: {predict_file}")
    click.echo(f"⚗️  Fields: {fields} ({', '.join(FIELD_OPTIONS[fields])})")
    click.echo(f"🔬 Grid resolution: {grid_resolution}")
    click.echo(f"📊 Components: {num_components}")
    
    if not VISUALIZATION_AVAILABLE:
        click.echo("⚠️  Warning: Visualization dependencies not available. Install pyvista for 3D visualization.")
    
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load data from SDF files
    click.echo("📂 Loading molecular data...")
    
    if predict_file is not None:
        # Load both training and prediction files
        predict_smiles_list, predict_mols, _ = data_loader.load_sdf_data(
            predict_file, is_training=False
        )
        train_smiles_list, train_mols, train_activities = data_loader.load_sdf_data(
            train_file, sdf_activity, is_training=True
        )
        # Combine molecules (assuming pre-aligned)
        aligned_mols = train_mols + predict_mols
        click.echo(f"✅ Loaded {len(train_mols)} training molecules and {len(predict_mols)} prediction molecules")
    else:
        # Load only training file
        train_smiles_list, train_mols, train_activities = data_loader.load_sdf_data(
            train_file, sdf_activity, is_training=True
        )
        aligned_mols = train_mols
        click.echo(f"✅ Loaded {len(train_mols)} training molecules")
    
    # Extract training molecules for further processing
    train_aligned_mols = [mol for mol, is_training in aligned_mols if is_training]
    
    # Initialize calculators and visualizers
    grid_calculator = MolecularGridCalculator()
    field_calculator = MolecularFieldCalculator()
    if VISUALIZATION_AVAILABLE:
        visualizer = MolecularVisualizer()
        contour_visualizer = ContourPlotVisualizer()
    
    # Calculate grid
    click.echo("🔲 Calculating molecular grid...")
    grid_spacing, grid_dimensions, grid_origin = grid_calculator.generate_grid(
        aligned_mols, grid_resolution, grid_padding
    )
    click.echo(f"✅ Grid calculated: {grid_dimensions} points with {grid_resolution} Å spacing")
    
    # Calculate fields
    click.echo("⚗️  Calculating molecular fields...")
    all_field_values = field_calculator.calc_field(
        aligned_mols, grid_spacing, grid_dimensions, grid_origin
    )
    click.echo("✅ Molecular fields calculated")
    
    # Visualize fields (if visualization is enabled and available)
    if not disable_visualization and VISUALIZATION_AVAILABLE:
        click.echo("📊 Generating field visualizations...")
        visual_field = all_field_values['train_fields']
        
        # Create field dictionary for the first molecule using selected fields
        first_molecule_fields = {
            field: visual_field[f"{field}_field"][0] for field in FIELD_OPTIONS[fields]
        }
        
        visualizer.visualize_field(
            train_aligned_mols[0],
            grid_spacing, grid_dimensions, grid_origin,
            first_molecule_fields, output_dir
        )
        click.echo("✅ Field visualizations saved")
    elif not disable_visualization and not VISUALIZATION_AVAILABLE:
        click.echo("⚠️  Visualization requested but dependencies not available")
    else:
        click.echo("⏭️  Visualization disabled, skipping field plots")
    
    # Generate contour plots (if visualization is enabled)
    if not disable_visualization and VISUALIZATION_AVAILABLE:
        click.echo("🗺️  Generating contour plots...")
        
        # Use raw field values for contour visualization (skipping PLS for now)
        train_field_values = all_field_values['train_fields']
        
        # Average field values across all molecules using selected fields
        all_molecule_field_values = {}
        for field in FIELD_OPTIONS[fields]:
            field_key = f"{field}_field"
            # Stack all molecule field values and take the mean across molecules
            stacked_fields = np.array([field_data.reshape(grid_dimensions) 
                                     for field_data in train_field_values[field_key]])
            all_molecule_field_values[field_key] = np.mean(stacked_fields, axis=0)
        
        # Calculate significant ranges for contour visualization
        significant_ranges = contour_visualizer.calculate_significant_ranges(
            all_molecule_field_values
        )
        
        # Generate contour plots using averaged field values
        contour_visualizer.visualize_contour_plots(
            train_aligned_mols[0], 
            all_molecule_field_values,
            grid_dimensions, grid_origin, grid_spacing,
            output_dir, significant_ranges
        )
        click.echo("✅ Contour plots saved")
    elif not disable_visualization and not VISUALIZATION_AVAILABLE:
        click.echo("⚠️  Contour plots requested but dependencies not available")
    else:
        click.echo("⏭️  Contour plots disabled")
    
    click.echo("✨ PyCoMSIA analysis complete!")
    

if __name__ == '__main__':
    main()