import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(results_dfs, method_names, true_signals_df, pvalue_col_names, alpha=0.05, 
                      output_csv=None, seed=None, simulation_index=None, all_true_positives=True):
    """
    Calculate statistical metrics for simulation results, both overall and per cell type.
    
    Parameters:
    -----------
    results_dfs : list of pandas.DataFrame
        List of dataframes containing simulation results.
    method_names : list of str
        Names of the methods corresponding to each dataframe.
    true_signals_df : pandas.DataFrame
        Dataframe containing the true signal settings.
    pvalue_col_names : list of str
        Column names of p-values to use for each method.
    alpha : float, optional
        Significance threshold. Default is 0.05.
    output_csv : str, optional
        Path to save the results CSV. If None or empty string, results won't be saved to disk.
    seed : int, optional
        Random seed used for the simulation.
    simulation_index : int or str, optional
        Index or identifier for the simulation.
    all_true_positives : bool, optional
        If True, calculate power using all true positives (even those not tested).
        If False (default), calculate power using only tested true positives.
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing the calculated metrics per cell type and overall.
    """
    # Initialize results dictionary
    results = {
        'Simulation_Index': [],
        'Seed': [],
        'Method': [],
        'Celltype': [],
        'TP': [],
        'TN': [],
        'FP': [],
        'FN': [],
        'Power': [],
        'Type_I_Error': [],
        'FDR': []
    }
    
    # Create a set of true positive gene-celltype pairs
    true_positives_set = set(zip(true_signals_df['gene_id'], true_signals_df['celltype']))
    
    # Process each method
    for idx, (df, method_name, pvalue_col) in enumerate(zip(results_dfs, method_names, pvalue_col_names)):
        # Make sure column names are consistent
        gene_col = 'Gene' if 'Gene' in df.columns else 'gene_id'
        celltype_col = 'Celltype' if 'Celltype' in df.columns else 'celltype'
        
        # Get all cell types in the data
        cell_types = df[celltype_col].unique()
        
        # First, calculate metrics for each cell type
        for cell_type in cell_types:
            # Filter data for current cell type
            cell_type_df = df[df[celltype_col] == cell_type]
            
            # Create a set of gene-celltype pairs that are called significant
            significant_pairs = set(zip(
                cell_type_df.loc[cell_type_df[pvalue_col] < alpha, gene_col],
                cell_type_df.loc[cell_type_df[pvalue_col] < alpha, celltype_col]
            ))
            
            # Get all gene-celltype pairs tested for this cell type
            all_tested_pairs = set(zip(cell_type_df[gene_col], cell_type_df[celltype_col]))
            
            # True positive pairs for this cell type
            true_positives_this_celltype = {pair for pair in true_positives_set if pair[1] == cell_type}
            
            # Find the intersection of tested pairs and true positive pairs
            tested_true_positives = true_positives_this_celltype.intersection(all_tested_pairs)
            
            # Calculate metrics
            # Power calculation depends on the all_true_positives parameter
            if all_true_positives:
                # Use all true positives in the denominator, regardless of whether they were tested
                TP = len(significant_pairs.intersection(true_positives_this_celltype))
                FP = len(significant_pairs - true_positives_this_celltype)
                FN = len(true_positives_this_celltype - significant_pairs)
                TN = len(all_tested_pairs) - TP - FP - FN
                power = TP / len(true_positives_this_celltype) if len(true_positives_this_celltype) > 0 else 0
            else:
                # Use only tested true positives in the denominator (default)
                TP = len(significant_pairs.intersection(tested_true_positives))
                FP = len(significant_pairs - tested_true_positives)
                FN = len(tested_true_positives - significant_pairs)
                TN = len(all_tested_pairs) - TP - FP - FN
                power = TP / len(tested_true_positives) if len(tested_true_positives) > 0 else 0
                
            type_I_error = FP / (FP + TN) if (FP + TN) > 0 else 0
            fdr = FP / (FP + TP) if (FP + TP) > 0 else 0
            
            # Append to results
            results['Method'].append(method_name)
            results['Celltype'].append(cell_type)
            results['TP'].append(TP)
            results['TN'].append(TN)
            results['FP'].append(FP)
            results['FN'].append(FN)
            results['Power'].append(power)
            results['Type_I_Error'].append(type_I_error)
            results['FDR'].append(fdr)
            results['Seed'].append(seed)
            results['Simulation_Index'].append(simulation_index)
        
        # Then, calculate overall metrics across all cell types
        
        # Create a set of gene-celltype pairs that are called significant
        significant_pairs = set(zip(
            df.loc[df[pvalue_col] < alpha, gene_col],
            df.loc[df[pvalue_col] < alpha, celltype_col]
        ))
        
        # Get all gene-celltype pairs tested
        all_tested_pairs = set(zip(df[gene_col], df[celltype_col]))
        
        # Find the intersection of tested pairs and true positive pairs
        tested_true_positives = true_positives_set.intersection(all_tested_pairs)
        
        # Calculate metrics
        # Calculate derived metrics - for overall metrics
        if all_true_positives:
            # Use all true positives in the denominator
            TP = len(significant_pairs.intersection(true_positives_set))
            FP = len(significant_pairs - true_positives_set)
            FN = len(true_positives_set - significant_pairs)
            TN = len(all_tested_pairs) - TP - FP - FN
            power = TP / len(true_positives_set) if len(true_positives_set) > 0 else 0
        else:
            # Use only tested true positives in the denominator
            TP = len(significant_pairs.intersection(tested_true_positives))
            FP = len(significant_pairs - tested_true_positives)
            FN = len(tested_true_positives - significant_pairs)
            TN = len(all_tested_pairs) - TP - FP - FN
            power = TP / len(tested_true_positives) if len(tested_true_positives) > 0 else 0
            
        type_I_error = FP / (FP + TN) if (FP + TN) > 0 else 0
        fdr = FP / (FP + TP) if (FP + TP) > 0 else 0
        
        # Append to results
        results['Method'].append(method_name)
        results['Celltype'].append('overall')
        results['TP'].append(TP)
        results['TN'].append(TN)
        results['FP'].append(FP)
        results['FN'].append(FN)
        results['Power'].append(power)
        results['Type_I_Error'].append(type_I_error)
        results['FDR'].append(fdr)
        results['Seed'].append(seed)
        results['Simulation_Index'].append(simulation_index)
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    results_df['Method'] = pd.Categorical(
        results_df['Method'],
        categories=method_names,
        ordered=True
    )
    
    # Save to CSV if output_csv is provided
    if output_csv:
        results_df.to_csv(output_csv, index=False)
    
    return results_df

def plot_metrics_comparison_barplot(metrics_df, metric='Power', ax=None, save_path=None, figsize=(12, 6), 
                          celltypes=None, group_by='celltype', title=None, ylim=None, seed=None,
                          method_order=None):
    """
    Create a bar plot comparing metrics (Power, Type_I_Error, FDR, etc.) across different methods for multiple cell types.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        Dataframe containing the calculated metrics from calculate_metrics function.
    metric : str, optional
        Name of the metric to plot (column name in metrics_df). Default is 'Power'.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure will be created.
    save_path : str, optional
        Path to save the figure. If None or empty string, figure won't be saved to disk.
    figsize : tuple, optional
        Figure size. Default is (12, 6). Only used if ax is None.
    celltypes : list of str, optional
        List of cell types to include in the plot. If None, uses all cell types.
    group_by : str, optional
        How to group the bars. Options are 'method' or 'celltype' (default).
        'method' groups bars by method with different colors for cell types.
        'celltype' groups bars by cell type with different colors for methods.
    title : str, optional
        Custom title for the plot. If None, a default title will be generated.
    ylim : tuple, optional
        Control the y-axis limits as (ymin, ymax). If None, automatically determined.
    seed : int, optional
        Specific seed to plot. If None, uses all seeds and adds error bars.
    method_order : list, optional
        Order of methods to use for plotting. If None, uses order from the DataFrame.
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object containing the plot.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Make a copy of the input dataframe to avoid modifying it
    plot_metrics_df = metrics_df.copy()
    
    # Set method order if provided
    if method_order is not None:
        plot_metrics_df['Method'] = pd.Categorical(
            plot_metrics_df['Method'],
            categories=method_order,
            ordered=True
        )
    
    # Default to all celltypes if no celltypes specified
    if celltypes is None:
        celltypes = np.unique(plot_metrics_df['Celltype']).tolist()
    
    # Filter data for the specified cell types
    filtered_df = plot_metrics_df[plot_metrics_df['Celltype'].isin(celltypes)]
    
    # Handle seed filtering
    if seed is not None:
        # Filter for the specific seed
        plot_df = filtered_df[filtered_df['Seed'] == seed].copy()
        use_errorbar = False
    else:
        # Use all seeds and prepare for error bars
        use_errorbar = True
        
        # Create aggregation by Method and Celltype to get mean and std for each group
        if group_by.lower() == 'method':
            agg_groups = ['Method', 'Celltype']
        else:
            agg_groups = ['Celltype', 'Method']
        
        # Calculate mean and standard error for each group
        plot_df = filtered_df.groupby(agg_groups, observed=False)[metric].agg(['mean', 'std']).reset_index()
        plot_df.rename(columns={'mean': metric}, inplace=True)
        
        # Calculate standard error from standard deviation
        n_seeds = len(filtered_df['Seed'].unique())
        if n_seeds > 1:
            plot_df['se'] = plot_df['std'] / np.sqrt(n_seeds)
        else:
            plot_df['se'] = 0
    
    # Create a new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Determine plotting approach based on group_by
    if use_errorbar:
        # Plot with error bars
        if group_by.lower() == 'method':
            sns.barplot(x='Method', y=metric, hue='Celltype', data=plot_df, ax=ax, 
                         errorbar=('se', 1), err_kws={'linewidth': 1.5}, order=plot_df['Method'].cat.categories)
        else:
            sns.barplot(x='Celltype', y=metric, hue='Method', data=plot_df, ax=ax, 
                         errorbar=('se', 1), err_kws={'linewidth': 1.5}, hue_order=plot_df['Method'].cat.categories)
    else:
        # Standard plot without error bars
        if group_by.lower() == 'method':
            sns.barplot(x='Method', y=metric, hue='Celltype', data=plot_df, ax=ax,
                        order=plot_df['Method'].cat.categories)
        else:
            sns.barplot(x='Celltype', y=metric, hue='Method', data=plot_df, ax=ax,
                        hue_order=plot_df['Method'].cat.categories)
    
    # Add labels and title
    if group_by.lower() == 'method':
        ax.set_xlabel('Method', fontsize=12)
    else:
        ax.set_xlabel('Cell Type', fontsize=12)
    
    ax.set_ylabel(metric, fontsize=12)
    
    # Use custom title if provided, otherwise create one based on metric and cell types
    if title is not None:
        plot_title = title
    else:
        if len(celltypes) == 1:
            plot_title = f'{metric} Comparison - {celltypes[0].capitalize()}'
        else:
            plot_title = f'{metric} Comparison Across Cell Types'
    
    ax.set_title(plot_title, fontsize=14)
    
    # Add legend with better positioning
    ax.legend(title='Cell Type' if group_by.lower() == 'method' else 'Method', 
              loc='upper right', bbox_to_anchor=(1, 1))
    
    # Set y-axis limits if provided, otherwise adjust based on data
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # Get the maximum value for the metric
        max_val = plot_df[metric].max() if not plot_df.empty else 0
        
        # For error bars, add the error amount to max_val
        if use_errorbar and 'se' in plot_df.columns:
            max_val += plot_df['se'].max() * 2  # Add twice the max standard error
            
        # Set y-axis limits with some padding
        ax.set_ylim(0, min(1.0, max_val * 1.15 if max_val > 0 else 0.1))
    
    # Rotate x-axis labels if needed
    primary_dim = 'Method' if group_by.lower() == 'method' else 'Celltype'
    if len(plot_df[primary_dim].unique()) > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on top of each bar
    if not use_errorbar or len(plot_df) <= 10:  # Only add labels if not too crowded
        bars = ax.patches
        
        # Get offsets for grouped bars
        n_groups = len(plot_df[primary_dim].unique())
        n_items = len(plot_df['Method' if group_by.lower() == 'celltype' else 'Celltype'].unique())
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            # Format based on metric (percentages for metrics that range from 0 to 1)
            if metric in ['Power', 'Type_I_Error', 'FDR'] and height <= 1:
                value_text = f'{height:.3f}'
            else:
                value_text = f'{height:.1f}'
                
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                   value_text, ha='center', va='bottom', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return ax

def plot_metrics_comparison_boxplot(metrics_df, metric='Power', ax=None, save_path=None, figsize=(12, 6), 
                                  celltypes=None, group_by='celltype', title=None, ylim=None, 
                                  method_order=None, show_points=True):
    """
    Create a box plot comparing metrics distributions across different methods for multiple cell types.
    Falls back to bar plot with error bars if insufficient data for boxplots.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        Dataframe containing the calculated metrics from calculate_metrics function.
    metric : str, optional
        Name of the metric to plot (column name in metrics_df). Default is 'Power'.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure will be created.
    save_path : str, optional
        Path to save the figure. If None or empty string, figure won't be saved to disk.
    figsize : tuple, optional
        Figure size. Default is (12, 6). Only used if ax is None.
    celltypes : list of str, optional
        List of cell types to include in the plot. If None, uses all cell types.
    group_by : str, optional
        How to group the boxes. Options are 'method' or 'celltype' (default).
        'method' groups boxes by method with different colors for cell types.
        'celltype' groups boxes by cell type with different colors for methods.
    title : str, optional
        Custom title for the plot. If None, a default title will be generated.
    ylim : tuple, optional
        Control the y-axis limits as (ymin, ymax). If None, automatically determined.
    method_order : list, optional
        Order of methods to use for plotting. If None, uses order from the DataFrame.
    show_points : bool, optional
        Whether to show individual data points on the box plot. Default is True.
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object containing the plot.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Make a copy of the input dataframe to avoid modifying it
    plot_metrics_df = metrics_df.copy()
    
    # Set method order if provided
    if method_order is not None:
        plot_metrics_df['Method'] = pd.Categorical(
            plot_metrics_df['Method'],
            categories=method_order,
            ordered=True
        )
    
    # Default to all celltypes if no celltypes specified
    if celltypes is None:
        celltypes = np.unique(plot_metrics_df['Celltype']).tolist()
    
    # Filter data for the specified cell types
    filtered_df = plot_metrics_df[plot_metrics_df['Celltype'].isin(celltypes)]
    
    # Create a new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Determine plotting approach based on group_by
    if group_by.lower() == 'method':
        # Group by method, with different colors for cell types
        x_var, hue_var = 'Method', 'Celltype'
        order = filtered_df['Method'].cat.categories if hasattr(filtered_df['Method'], 'cat') else None
        hue_order = None
    else:
        # Group by cell type, with different colors for methods
        x_var, hue_var = 'Celltype', 'Method'
        order = None
        hue_order = filtered_df['Method'].cat.categories if hasattr(filtered_df['Method'], 'cat') else None
    
    # Create the boxplot
    sns.boxplot(x=x_var, y=metric, hue=hue_var, data=filtered_df, ax=ax,
                order=order, hue_order=hue_order, showfliers=False, width=0.6)
    
    # Add individual points if requested
    if show_points:
        sns.stripplot(x=x_var, y=metric, hue=hue_var, data=filtered_df, 
                     dodge=True, size=4, linewidth=0, alpha=0.6, ax=ax,
                     order=order, hue_order=hue_order, legend=False)
    
    # Add labels and title
    ax.set_xlabel(x_var, fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    
    # Use custom title if provided, otherwise create one based on metric and cell types
    if title is not None:
        plot_title = title
    else:
        if len(celltypes) == 1:
            plot_title = f'{metric} Distribution - {celltypes[0].capitalize()}'
        else:
            plot_title = f'{metric} Distribution Across Cell Types'
    
    ax.set_title(plot_title, fontsize=14)
    
    # Add legend with better positioning
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        # Take only first set of handles/labels if there are duplicates
        if len(handles) > len(filtered_df[hue_var].unique()):
            handles = handles[:len(filtered_df[hue_var].unique())]
            labels = labels[:len(filtered_df[hue_var].unique())]
            
        ax.legend(handles, labels, 
                 title=hue_var, 
                 loc='upper right', 
                 bbox_to_anchor=(1, 1))
    
    # Set y-axis limits if provided, otherwise adjust based on data
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # Get the maximum value for the metric
        max_val = filtered_df[metric].max() if not filtered_df.empty else 0
            
        # Set y-axis limits with some padding
        ax.set_ylim(0, min(1.0, max_val * 1.15 if max_val > 0 else 0.1))
    
    # Rotate x-axis labels if needed
    if len(filtered_df[x_var].unique()) > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Calculate and add median value labels directly using pyplot text
    # This approach is more reliable than trying to identify boxes
    if x_var == 'Method':
        for i, method in enumerate(filtered_df[x_var].unique()):
            for j, celltype in enumerate(filtered_df[hue_var].unique()):
                data = filtered_df[(filtered_df[x_var] == method) & (filtered_df[hue_var] == celltype)]
                if not data.empty:
                    median_val = data[metric].median()
                    
                    # Format the value
                    if metric in ['Power', 'Type_I_Error', 'FDR'] and median_val <= 1:
                        value_text = f'{median_val:.3f}'
                    else:
                        value_text = f'{median_val:.2f}'
                    
                    # Position for the text: x-coordinate accounting for dodge
                    width = 0.8 / len(filtered_df[hue_var].unique())
                    x_pos = i + width * (j - (len(filtered_df[hue_var].unique()) - 1) / 2)
                    
                    # Y-coordinate with small offset above the median
                    y_pos = median_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                    
                    ax.text(x_pos, y_pos, value_text, ha='center', va='bottom', fontsize=8)
    else:  # x_var == 'Celltype'
        for i, celltype in enumerate(filtered_df[x_var].unique()):
            for j, method in enumerate(filtered_df[hue_var].unique()):
                data = filtered_df[(filtered_df[x_var] == celltype) & (filtered_df[hue_var] == method)]
                if not data.empty:
                    median_val = data[metric].median()
                    
                    # Format the value
                    if metric in ['Power', 'Type_I_Error', 'FDR'] and median_val <= 1:
                        value_text = f'{median_val:.3f}'
                    else:
                        value_text = f'{median_val:.2f}'
                    
                    # Position for the text: x-coordinate accounting for dodge
                    width = 0.8 / len(filtered_df[hue_var].unique())
                    x_pos = i + width * (j - (len(filtered_df[hue_var].unique()) - 1) / 2)
                    
                    # Y-coordinate with small offset above the median
                    y_pos = median_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                    
                    ax.text(x_pos, y_pos, value_text, ha='center', va='bottom', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return ax


def plot_metrics_comparison_linechart(metrics_df, metric='Power', ax=None, save_path=None, figsize=(12, 6), 
                                    celltype='overall', title=None, ylim=None, method_order=None,
                                    index_list=None, connect_points=True, marker_size=8):
    """
    Create a line plot showing metric trends across different simulation indices for each method.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        Dataframe containing the calculated metrics from calculate_metrics function.
    metric : str, optional
        Name of the metric to plot (column name in metrics_df). Default is 'Power'.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure will be created.
    save_path : str, optional
        Path to save the figure. If None or empty string, figure won't be saved to disk.
    figsize : tuple, optional
        Figure size. Default is (12, 6). Only used if ax is None.
    celltype : str, optional
        Cell type to include in the plot. Default is 'overall'.
    title : str, optional
        Custom title for the plot. If None, a default title will be generated.
    ylim : tuple, optional
        Control the y-axis limits as (ymin, ymax). If None, automatically determined.
    method_order : list, optional
        Order of methods to use for plotting. If None, uses order from the DataFrame.
    index_list : list, optional
        List of simulation indices to include in the plot, in the order they should appear.
        If None, all indices in the data will be used.
    connect_points : bool, optional
        Whether to connect points with lines. Default is True. If False, shows only markers.
    marker_size : int, optional
        Size of markers. Default is 8.
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object containing the plot.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Make a copy of the input dataframe to avoid modifying it
    plot_metrics_df = metrics_df.copy()
    
    # Set method order if provided
    if method_order is not None:
        plot_metrics_df['Method'] = pd.Categorical(
            plot_metrics_df['Method'],
            categories=method_order,
            ordered=True
        )
    
    # Filter data for the specified celltype
    filtered_df = plot_metrics_df[plot_metrics_df['Celltype'] == celltype]
    
    # Filter by simulation indices if index_list is provided
    if index_list is not None:
        # Convert all indices to the same type as in index_list
        if all(isinstance(idx, (int, float)) for idx in index_list):
            try:
                filtered_df['Simulation_Index'] = pd.to_numeric(filtered_df['Simulation_Index'])
            except:
                pass
        
        # Keep only the indices in the list
        filtered_df = filtered_df[filtered_df['Simulation_Index'].isin(index_list)]
        
        # Set categorical order based on index_list
        filtered_df['Simulation_Index'] = pd.Categorical(
            filtered_df['Simulation_Index'],
            categories=index_list,
            ordered=True
        )
    else:
        # If no index_list, try to convert to numeric for natural sorting
        try:
            filtered_df['Simulation_Index'] = pd.to_numeric(filtered_df['Simulation_Index'])
            filtered_df = filtered_df.sort_values('Simulation_Index')
        except:
            # If not numeric, keep as is
            pass
    
    # Calculate mean and standard error for each simulation index and method
    agg_df = filtered_df.groupby(['Simulation_Index', 'Method'], observed=False)[metric].agg(['mean', 'std']).reset_index()
    agg_df.rename(columns={'mean': metric}, inplace=True)
    
    # Calculate standard error from standard deviation
    n_seeds = len(filtered_df['Seed'].unique())
    if n_seeds > 1:
        agg_df['se'] = agg_df['std'] / np.sqrt(n_seeds)
    else:
        agg_df['se'] = 0
    
    # Create a new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique methods in order
    methods = agg_df['Method'].unique()
    
    # Get unique simulation indices in order
    if index_list is not None:
        # Use only indices that exist in the data
        simulation_indices = [idx for idx in index_list if idx in agg_df['Simulation_Index'].values]
    else:
        simulation_indices = sorted(agg_df['Simulation_Index'].unique())
    
    # Set up colors
    colors = plt.cm.tab10.colors
    
    # Plot each method as a line with error bars
    for i, method in enumerate(methods):
        method_data = agg_df[agg_df['Method'] == method]
        
        # Create a new dataframe with all simulation indices for this method
        plot_data = []
        for j, idx in enumerate(simulation_indices):
            idx_data = method_data[method_data['Simulation_Index'] == idx]
            if not idx_data.empty:
                plot_data.append({
                    'Method': method,
                    'Simulation_Index': idx,
                    'x_pos': j,
                    metric: idx_data[metric].values[0],
                    'se': idx_data['se'].values[0]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        if not plot_df.empty:
            # Plot the line
            line_style = '-' if connect_points else ''
            ax.errorbar(plot_df['x_pos'], plot_df[metric], yerr=plot_df['se'], 
                       fmt=f'{line_style}o', markersize=marker_size, 
                       label=method, color=colors[i % len(colors)],
                       capsize=5, elinewidth=1.5, markeredgecolor='white')
    
    # Set x-axis ticks and labels
    ax.set_xticks(np.arange(len(simulation_indices)))
    ax.set_xticklabels(simulation_indices)
    
    # Add labels and title
    ax.set_xlabel('Simulation Index', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    
    # Use custom title if provided, otherwise create one based on metric
    if title is not None:
        plot_title = title
    else:
        plot_title = f'{metric} Trend Across Simulations - {celltype.capitalize()}'
    
    ax.set_title(plot_title, fontsize=14)
    
    # Add legend
    ax.legend(title='Method', loc='best')
    
    # Set y-axis limits if provided, otherwise adjust based on data
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # Calculate appropriate y limits
        y_min = max(0, agg_df[metric].min() - 2 * agg_df['se'].max())
        
        if metric in ['Power', 'Type_I_Error', 'FDR']:
            # For percentage metrics, cap at 1.0
            y_max = min(1.0, agg_df[metric].max() + 2 * agg_df['se'].max())
        else:
            y_max = agg_df[metric].max() + 2 * agg_df['se'].max()
            
        # Add some padding
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    if len(methods) * len(simulation_indices) <= 10:  # Only if not too crowded
        for i, method in enumerate(methods):
            method_points = []
            
            # Collect all points for this method
            for j, idx in enumerate(simulation_indices):
                method_idx_data = agg_df[(agg_df['Method'] == method) & (agg_df['Simulation_Index'] == idx)]
                if not method_idx_data.empty:
                    method_points.append({
                        'x_pos': j,
                        'y_val': method_idx_data[metric].values[0],
                        'se': method_idx_data['se'].values[0]
                    })
            
            # Add labels for each point
            for point in method_points:
                # Format based on metric
                if metric in ['Power', 'Type_I_Error', 'FDR'] and point['y_val'] <= 1:
                    value_text = f'{point["y_val"]:.3f}'
                else:
                    value_text = f'{point["y_val"]:.2f}'
                
                # Position the label above the point
                y_pos = point['y_val'] + point['se'] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                ax.annotate(value_text, (point['x_pos'], y_pos), ha='center', va='bottom', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return ax
