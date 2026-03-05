import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


class DataLoader:
    def __init__(self):
        self.smiles_list = None
        self.activities = None
        self.molecules = None
        
    def load_sdf_data(self, sdf_file, activity_property=None, is_training=True):
        """
        Load SMILES, 3D molecules, and activity data from SDF

        Parameters:
        -----------
        sdf_file : str
            Path to the SDF file
        activity_property : str, optional
            Name of the property in the SDF file that contains the activity data
            Required if is_training=True
        is_training : bool, default=True
            If True, loads both SMILES and activities
            If False, loads only SMILES (for prediction dataset)

        Returns:
        --------
        tuple
            If is_training=True: (smiles_list, [(mol, is_training)], activities)
            If is_training=False: (smiles_list, [(mol, is_training)], None)
        """
        # Read the SDF file
        print("Reading SDF file...")
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        self.smiles_list = []
        mols_with_flag = []  # List of tuples: (mol, is_training)
        self.activities = []

        for mol in suppl:
            if mol is not None:
                # Ensure all molecules have hydrogens (add if missing, keep if present)
                mol_with_hs = Chem.AddHs(mol)
                
                # Generate SMILES from the molecule with hydrogens
                smiles = Chem.MolToSmiles(mol_with_hs)
                self.smiles_list.append(smiles)

                # Store the 3D molecule with hydrogens and the is_training flag
                mols_with_flag.append((mol_with_hs, is_training))

                # Extract activity if in training mode (from original mol to preserve properties)
                if is_training:
                    if activity_property is None:
                        raise ValueError("Activity property must be specified for training data")
                    if activity_property not in mol.GetPropNames():
                        raise ValueError(f"Activity property '{activity_property}' not found in SDF file")
                    activity = float(mol.GetProp(activity_property))
                    self.activities.append(activity)

        if not is_training:
            self.activities = None

        return self.smiles_list, mols_with_flag, self.activities