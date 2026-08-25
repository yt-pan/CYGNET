import numpy as np
from statsmodels.stats.multitest import multipletests
import os
import pandas as pd
from glob import glob


def p_value_adjust_with_permutation_ecdf(p_values, permuted_p_values):
    """
    Adjusts p-values using the empirical cumulative distribution function (ECDF) of permuted p-values.
    
    Parameters:
    -----------
    p_values : array-like
        Original p-values to be adjusted
    permuted_p_values : array-like
        Reference p-values from permutation tests
        
    Returns:
    --------
    adjusted_p_values : numpy.ndarray
        Adjusted p-values based on the ECDF of permuted p-values
        
    Raises:
    -------
    ValueError
        If any p-value is outside [0, 1] or if permuted_p_values is empty
    """
    
    # Convert inputs to numpy arrays
    p_values = np.asarray(p_values)
    permuted_p_values = np.asarray(permuted_p_values)
    
    # Check if all p-values are valid (between 0 and 1)
    if np.any((p_values < 0) | (p_values > 1)) or np.any((permuted_p_values < 0) | (permuted_p_values > 1)):
        raise ValueError("All p-values must be between 0 and 1")
    
    # Check if permuted_p_values is empty
    if len(permuted_p_values) == 0:
        raise ValueError("permuted_p_values cannot be empty")
    
    # Sort permuted p-values and calculate ECDF
    sorted_perm = np.sort(permuted_p_values)
    ecdf = np.arange(1, len(sorted_perm) + 1) / len(sorted_perm)
    
    # Add boundary values (0 and 1) to sorted permuted p-values and ECDF
    sorted_perm = np.concatenate([[0], sorted_perm, [1]])
    ecdf = np.concatenate([[0], ecdf, [1]])
    
    # Interpolate original p-values using the ECDF of permuted p-values
    adjusted_p_values = np.interp(p_values, sorted_perm, ecdf)
    
    return adjusted_p_values

def fdr_control(p_values, method='fdr_bh', permutation=False, permuted_p_values=None):
    """
    Adjusts p-values using the specified method for controlling the false discovery rate (FDR).
    
    Parameters:
    -----------
    p_values : array-like
        Original p-values to be adjusted
    method : str, optional
        Method for controlling the false discovery rate (FDR). Possible values are:
            - 'fdr_bh' (default): Benjamini-Hochberg procedure
            - 'fdr_by': Benjamini-Yekutieli procedure
            - 'fdr_tsbh': Two-stage Benjamini-Hochberg procedure
            - 'fdr_tsbky': Two-stage Benjamini-Krieger-Yekutieli procedure
    permutation : bool, optional
        Whether to use permutation-based p-values for adjustment
    permuted_p_values : array-like, optional
        Reference p-values from permutation tests
        
    Returns:
    --------
    adjusted_p_values : numpy.ndarray
        Adjusted p-values based on the specified FDR control method
    p_values : numpy.ndarray
        Original p-values (unchanged if permutation is False
        or adjusted using permutation-based ECDF if permutation is True)
        
    Raises:
    -------
    ValueError
        If any p-value is outside [0, 1] or if permuted_p_values is empty
    """
    
    p_values = np.asarray(p_values)
    
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("All p-values must be between 0 and 1")
    
    if permutation:
        if permuted_p_values is None:
            raise ValueError("permuted_p_values cannot be None")
        
        p_values = p_value_adjust_with_permutation_ecdf(p_values, permuted_p_values)
    
    else:
        if permuted_p_values is not None:
            print("Warning: permuted_p_values will be ignored since permutation is set to False")

    adjusted_p_values = multipletests(p_values, method=method)[1]
    
    return adjusted_p_values, p_values


def run_celltype_fdr_analysis(real_file, permu_file, prefix='fdr_result', output_dir='', method='fdr_bh', alpha=0.05, ignore_missing=True,
                              overwrite=False):
    """
    Analyzes gene expression data for all cell types with FDR control.
    
    Parameters:
    -----------
    real_file : str
        Path for CSV file containing real p-values for all cell types
    permu_file : str
        Path for CSV file containing permuted p-values for all cell types
    prefix : str
        Prefix for output files
    output_dir : str
        Directory for output files
    method : str, optional
        Method for FDR control (default: 'fdr_bh'), should be one of accepted methods in statsmodels.stats.multitest.multipletests
    alpha : float, optional
        Significance threshold for filtering genes in per-cell type files
    """

    # File paths
    if not output_dir:
        output_dir = os.path.dirname(real_file)
    os.makedirs(output_dir, exist_ok=True)
    out_file = f"{prefix}_all_results.csv"
    output_file = os.path.join(output_dir, out_file)

    if (not overwrite) and os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Use 'overwrite=True' to overwrite it.")
        all_results_df=pd.read_csv(output_file)
        return all_results_df
    
    # Read files
    try:
        df_r = pd.read_csv(real_file)
    except FileNotFoundError:
        if ignore_missing:
            print(f"Warning: Real file not found, skipping analysis: {real_file}")
            return None
        else:
            raise FileNotFoundError(f"Real file not found: {real_file}")
    
    # Try to read permutation file, handle if it doesn't exist
    try:
        df_p = pd.read_csv(permu_file)
        has_permutation = True
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: Permutation file not found or empty: {permu_file}")
        has_permutation = False
        df_p = None
    
    # Check required columns
    required_cols = ['Gene', 'Celltype', 'p_value']
    if not all(col in df_r.columns for col in required_cols):
        raise ValueError(f"Results file must contain columns: {', '.join(required_cols)}")
    if has_permutation and not all(col in df_p.columns for col in required_cols):
        raise ValueError(f"Permutation file must contain columns: {', '.join(required_cols)}")

    # Clean up p_values in the real data - filter out NAs and empty values
    initial_count = len(df_r)
    df_r = df_r[~pd.isna(df_r['p_value'])]  # Remove NaN values
    df_r = df_r[df_r['p_value'] != ""]  # Remove empty strings
    
    # Convert any string p-values to float
    try:
        df_r['p_value'] = df_r['p_value'].astype(float)
    except ValueError:
        print("Warning: Non-numeric values found in p_value column. Filtering them out.")
        df_r = df_r[pd.to_numeric(df_r['p_value'], errors='coerce').notna()]
        df_r['p_value'] = df_r['p_value'].astype(float)
    
    # Check for invalid p-values (outside 0-1 range)
    df_r = df_r[(df_r['p_value'] >= 0) & (df_r['p_value'] <= 1)]
    
    cleaned_count = len(df_r)
    if initial_count != cleaned_count:
        print(f"Filtered out {initial_count - cleaned_count} rows with invalid p-values")
    
    # If permutation data exists, clean it too
    if has_permutation:
        initial_perm_count = len(df_p)
        df_p = df_p[~pd.isna(df_p['p_value'])]  # Remove NaN values
        df_p = df_p[df_p['p_value'] != ""]  # Remove empty strings
        
        # Convert any string p-values to float
        try:
            df_p['p_value'] = df_p['p_value'].astype(float)
        except ValueError:
            print("Warning: Non-numeric values found in permutation p_value column. Filtering them out.")
            df_p = df_p[pd.to_numeric(df_p['p_value'], errors='coerce').notna()]
            df_p['p_value'] = df_p['p_value'].astype(float)
        
        # Check for invalid p-values (outside 0-1 range)
        df_p = df_p[(df_p['p_value'] >= 0) & (df_p['p_value'] <= 1)]
        
        cleaned_perm_count = len(df_p)
        if initial_perm_count != cleaned_perm_count:
            print(f"Filtered out {initial_perm_count - cleaned_perm_count} rows with invalid p-values from permutation data")
    
    # Get all unique cell types from the real data
    cell_types = df_r['Celltype'].unique()
    print(f"Found {len(cell_types)} cell types in the dataset")
    
    # Create an empty list to store results for all cell types
    all_results = []
    
    for ct in cell_types:
        print(f"Processing cell type: {ct}")
        
        # Filter data for current cell type
        df_r_ct = df_r[df_r['Celltype'] == ct]
        
        # Check if there are any valid p-values for this cell type
        if len(df_r_ct) == 0:
            print(f"No valid data found for cell type {ct}, skipping")
            continue
        
        observed_pvals = df_r_ct['p_value'].values
        gene_names = df_r_ct['Gene'].values
        
        # Additional check for valid p-values for this specific cell type
        if np.any(np.isnan(observed_pvals)) or len(observed_pvals) == 0:
            print(f"Warning: Found NaN values in p_values for {ct}, skipping this cell type")
            continue
        
        # Check if permutation data is available for this cell type
        if has_permutation:
            df_p_ct = df_p[df_p['Celltype'] == ct]
            if len(df_p_ct) > 0:
                permuted_pvals = df_p_ct['p_value'].values
                
                # Check for NaN or invalid values in permuted p-values
                if np.any(np.isnan(permuted_pvals)):
                    print(f"Warning: Found NaN values in permuted p_values for {ct}, proceeding without permutation adjustment")
                    permuted_pvals = None
                    has_permutation_for_ct = False
                else:
                    has_permutation_for_ct = True
            else:
                print(f"Warning: No permutation data found for cell type {ct}")
                permuted_pvals = None
                has_permutation_for_ct = False
        else:
            permuted_pvals = None
            has_permutation_for_ct = False
        
        # Initialize columns with NaN values
        perm_adjusted_pvals = np.full(len(observed_pvals), np.nan)
        fdr_pvals = np.full(len(observed_pvals), np.nan)
        fdr_perm_pvals = np.full(len(observed_pvals), np.nan)
        
        # FDR control on raw p-values
        try:
            fdr_pvals, _ = fdr_control(observed_pvals, method=method, permutation=False)
        except Exception as e:
            print(f"Error applying FDR control to raw p-values for {ct}: {e}")
        
        # Permutation adjustment and FDR control on permutation-adjusted p-values
        if has_permutation_for_ct:
            try:
                # Get permutation-adjusted p-values
                fdr_perm_pvals, perm_adjusted_pvals = fdr_control(observed_pvals, method=method, 
                                                    permutation=True, permuted_p_values=permuted_pvals)
            except Exception as e:
                print(f"Error applying permutation-based adjustment for {ct}: {e}")
        
        # Create a dataframe for this cell type
        ct_results = pd.DataFrame({
            'Gene': gene_names,
            'Celltype': ct,
            'Raw_p_value': observed_pvals,
            'Permutation_adjusted_p_value': perm_adjusted_pvals,
            'FDR_p_value': fdr_pvals,
            'FDR_permutation_adjusted_p_value': fdr_perm_pvals
        })
        
        # Add to the list of all results
        all_results.append(ct_results)
        
        # Create a separate file for this cell type with significant genes
        if has_permutation_for_ct:
            significant_genes = ct_results[ct_results['FDR_permutation_adjusted_p_value'] <= alpha].copy()
        else:
            significant_genes = ct_results[ct_results['FDR_p_value'] <= alpha].copy()
        
        if len(significant_genes) > 0:
            # Sort by adjusted p-value
            significant_genes = significant_genes.sort_values(by='Raw_p_value')

            # Save to file
            # if there is "/" in the cell type name, replace it with "_"
            ct = ct.replace("/", "_")
            ct_output_file = os.path.join(output_dir, f"{prefix}_{ct}_significant.csv")
            significant_genes.to_csv(ct_output_file, index=False)
            
            # Print top 10 (or fewer if less than 10 are significant)
            print(f"Top significant genes for {ct} (alpha={alpha}):")
            print(significant_genes.head(min(10, len(significant_genes))))
            print('\n')
        else:
            print(f"No significant genes found for {ct} at alpha={alpha}\n")
    
    # Check if we have any results
    if not all_results:
        print("Warning: No valid results found for any cell type")
        return None
    
    # Combine all results into a single dataframe
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by cell type and FDR-adjusted p-value (if available)
    all_results_df = all_results_df.sort_values(by=['FDR_permutation_adjusted_p_value', 'Celltype'], 
                                              na_position='last')
    
    # Save the combined results
    all_results_df.to_csv(output_file, index=False)
    
    print(f"All results saved to {output_file}")
    
    return all_results_df


def extract_significant_genes(results_file, output_dir, prefix, alpha=0.05):
    """
    Extracts significant genes from a results file and creates separate CSV files for each cell type.
    
    Parameters:
    -----------
    results_file : str
        Path to the CSV file containing all FDR analysis results
    output_dir : str
        Directory where significant gene files will be saved
    prefix : str
        Prefix for output filenames
    alpha : float, optional
        Significance threshold for filtering genes (default: 0.05)
        
    Returns:
    --------
    dict
        Dictionary mapping cell types to number of significant genes found
    """    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        results_df = pd.read_csv(results_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    # Check required columns
    required_cols = ['Gene', 'Celltype', 'FDR_p_value']
    if not all(col in results_df.columns for col in required_cols):
        raise ValueError(f"Results file must contain columns: {', '.join(required_cols)}")
    
    # Get all unique cell types
    cell_types = results_df['Celltype'].unique()
    print(f"Found {len(cell_types)} cell types in the dataset")
    
    # Dictionary to store counts of significant genes
    significant_counts = {}
    
    for ct in cell_types:
        # Filter data for current cell type
        ct_results = results_df[results_df['Celltype'] == ct].copy()
        
        # Check which p-value column to use (prefer permutation-adjusted if available)
        if 'FDR_permutation_adjusted_p_value' in ct_results.columns and not ct_results['FDR_permutation_adjusted_p_value'].isna().all():
            # Use permutation-adjusted FDR p-values
            significant_genes = ct_results[ct_results['FDR_permutation_adjusted_p_value'] <= alpha].copy()
            p_value_col = 'FDR_permutation_adjusted_p_value'
        else:
            # Fall back to regular FDR p-values
            significant_genes = ct_results[ct_results['FDR_p_value'] <= alpha].copy()
            p_value_col = 'FDR_p_value'
        
        # Store the count
        significant_counts[ct] = len(significant_genes)
        
        if len(significant_genes) > 0:
            # Sort by adjusted p-value
            significant_genes = significant_genes.sort_values(by=p_value_col)
            
            # Save to file
            ct_output_file = os.path.join(output_dir, f"{prefix}_{ct}_significant.csv")
            significant_genes.to_csv(ct_output_file, index=False)
            
            print(f"Found {len(significant_genes)} significant genes for {ct} at alpha={alpha}")
            print(f"Top genes:")
            print(significant_genes[['Gene', p_value_col]].head(min(10, len(significant_genes))))
            print(f"Saved to {ct_output_file}\n")
        else:
            print(f"No significant genes found for {ct} at alpha={alpha}\n")
    
    return significant_counts

def check_and_rearrange_dataframes(dfs, rearrange=False, only_common=False):
    """
    Check whether DataFrames have identical rows, potentially rearrange rows or 
    filter to common rows based on parameters.

    Args:
        dfs: List of pandas DataFrames to be checked.
        rearrange: Boolean indicating whether to rearrange DataFrames to have the same row order.
        only_common: Boolean indicating whether to filter DataFrames to only include common row names.

    Returns:
        A list of DataFrames that have been rearranged or filtered based on function arguments.
    """
    
    # Check if all DataFrames have the same row order
    reference_index = dfs[0].index
    if all(df.index.equals(reference_index) for df in dfs):
        print("All DataFrames have the same row order.")
        return dfs

    # Find common row names
    common_index = reference_index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)

    # If only_common is True, filter all DataFrames by the common index
    if only_common:
        if common_index.empty:
            raise ValueError("There are no common rows among the DataFrames.")
        print("Filtering DataFrames to have only common row names.")
        filtered_dfs = [df.loc[common_index] for df in dfs]
        return filtered_dfs

    # Check if all DataFrames have the same number of rows if only_common is not True
    row_numbers = [df.shape[0] for df in dfs]
    if not all(row_numbers[0] == rn for rn in row_numbers):
        raise ValueError("DataFrames do not have the same row number.")

    # Rearrange DataFrames if requested
    if rearrange:
        if common_index.empty:
            raise ValueError("There are no common rows to rearrange among the DataFrames.")
        print("DataFrames do not have the same row names. Rearranging DataFrames to have the same row order based on common row names.")
        rearranged_dfs = [df.loc[common_index] for df in dfs]
        print("DataFrames have been rearranged to have the same row order based on common row names.")
        return rearranged_dfs
    else:
        raise ValueError("DataFrames do not have the same row names. Set rearrange=True to rearrange DataFrames to have the same row order based on common row names.")

    return dfs


def load_simulation_data(simulation_dir, seed, normalized_type='sctransform', env_type='values'):
    """
    Load simulation data from a specific simulation directory and seed.
    
    Parameters
    ----------
    simulation_dir : str
        Path to the simulation directory (e.g., 'simulation_results/simulation_1')
    seed : int
        Seed number to load
    
    Returns
    -------
    tuple
        (locations_df, celltype_df, normalized_counts_df, env_values_df)
        - locations_df: DataFrame with x, y coordinates
        - celltype_df: DataFrame with cell IDs and cell types
        - normalized_counts_df: DataFrame with normalized gene expression counts
        - env_values_df: DataFrame with environmental values
    """
    # Initialize DataFrames
    locations_df = None
    celltype_df = None
    normalized_counts_df = None
    env_values_df = None
    
    # Get simulation index from directory name
    sim_index = os.path.basename(simulation_dir).split('_')[1]
    
    # Create file pattern for the specific seed
    pattern = f'sim_{sim_index}_{seed}_*.csv*'
    
    # Get all CSV files for this seed
    files = glob(os.path.join(simulation_dir, pattern))
    
    if not files:
        raise ValueError(f"No files found for simulation {sim_index}, seed {seed}")
    
    # Load each type of file
    for file in files:
        if 'locations' in file:
            locations_df = pd.read_csv(file, index_col=0)
        elif 'celltypes' in file:
            celltype_df = pd.read_csv(file, index_col=0)
        elif 'normalized_counts' in file and normalized_type in file:
            normalized_counts_df = pd.read_csv(file, index_col=0)
        elif 'env' in file and env_type in file:
            env_values_df = pd.read_csv(file, index_col=0)
            if env_type == 'values':
                # only select the first column if there are multiple
                # the other columns are not supposed to be used
                env_values_df = env_values_df.iloc[:, [0]]
    
    # change celltype_df to one-hot encoding
    celltype_df = pd.get_dummies(celltype_df, prefix='', prefix_sep='',dtype=float)

    # Check if all required data was loaded
    if any(df is None for df in [locations_df, celltype_df, normalized_counts_df, env_values_df]):
        missing = []
        if locations_df is None: missing.append("locations")
        if celltype_df is None: missing.append("celltypes")
        if normalized_counts_df is None: missing.append("normalized_counts")
        if env_values_df is None: missing.append("env_values")
        raise ValueError(f"Missing required data files: {', '.join(missing)}")
    
    return locations_df, celltype_df, normalized_counts_df, env_values_df
