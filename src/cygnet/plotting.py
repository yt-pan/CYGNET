import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os
import math
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d

def create_p_value_qq_plot(pvals_list, labels=None, s=10, title='', xlim=None, ylim=None, ax=None, figsize=(8, 8), show_plot=True):
    """
    Generate a QQ plot for multiple sets of p-values.

    Parameters:
    pvals_list : list of arrays or lists
        A list containing arrays or lists of p-values.
    labels : list of strings, optional
        List of labels for each set of p-values.
    s : int
        Marker size for the scatter plot.
    title : str, optional
        Title of the plot.
    xlim : tuple, optional
        Tuple of (min, max) for x-axis limits. If None, will be determined automatically.
    ylim : tuple, optional
        Tuple of (min, max) for y-axis limits. If None, will be determined automatically.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object to draw the plot onto. If None, a new figure will be created.
    figsize : tuple, optional
        Figure size in inches (width, height) if creating a new figure.
    
    Returns:
    matplotlib.axes.Axes
        The axes object containing the plot.
    """
    # Check if we need to create a new figure
    create_fig = ax is None
    
    if create_fig:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    n_sets = len(pvals_list)
    if labels is None:
        labels = [f'Set {i+1}' for i in range(n_sets)]
    
    for i, pval in enumerate(pvals_list):
        pval = np.array(pval)
        pval = pval[~np.isnan(pval)]  # Remove NA values
        n = len(pval)
        x = np.arange(1, n+1)
        
        # Calculate expected and empirical beta quantiles
        dat = pd.DataFrame({
            'obs': np.sort(pval),
            'exp': x / n,
            'upper': stats.beta.ppf(0.025, x, n-x+1),
            'lower': stats.beta.ppf(0.975, x, n-x+1)
        })

        # Transformations for plotting
        dat['log_exp'] = -np.log10(dat['exp'])
        dat['log_obs'] = -np.log10(dat['obs'])
        dat['log_upper'] = -np.log10(dat['upper'])
        dat['log_lower'] = -np.log10(dat['lower'])

        # Plot observed vs expected
        ax.scatter(dat['log_exp'], dat['log_obs'], s=s, label=labels[i], alpha=0.6)
        ax.plot(dat['log_exp'], dat['log_upper'], color='gray', linestyle='-', alpha=0.4)
        ax.plot(dat['log_exp'], dat['log_lower'], color='gray', linestyle='-', alpha=0.4)

    # Plot y=x line across the range
    max_log_exp = dat['log_exp'].max()
    max_log_obs = dat['log_obs'].max()
    line_range = min(max_log_exp, max_log_obs)
    ax.plot([0, line_range], [0, line_range], color='red', linestyle='-')

    # Set x and y limits
    if xlim is None:
        ax.set_xlim(0, np.ceil(max_log_exp))
    else:
        ax.set_xlim(xlim)
    
    if ylim is None:
        ax.set_ylim(0, np.ceil(max_log_obs))
    else:
        ax.set_ylim(ylim)

    # Customize plot appearance
    if title:
        ax.set_title(title)
    ax.set_xlabel(r"Expected $-\log_{10}$pv")
    ax.set_ylabel(r"Observed $-\log_{10}$pv")
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(False)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Only show the plot if we created a new figure
    if create_fig and show_plot:
        plt.show()
    
    return ax


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

def create_knn_pdp_plot(normalized_counts_df, celltype_df, location_df, env_values_df, 
                 genename, celltypename, envname, 
                 axes=None, plots_to_generate=(True, True, True), 
                 is_binary_celltype=False, n_neighbors=np.arange(20, 200, 10),
                 # parameters for styling
                 linewidth=2,
                 title_fontsize=14,
                 label_fontsize=12,
                 tick_fontsize=10,
                 legend_loc='best',
                 return_line_data=False):
    """
    Analyzes relationship between gene expression, cell type, and environment using KNN regression and PDP plots.
    
    Parameters:
    -----------
    normalized_counts_df : pandas DataFrame
        DataFrame containing normalized gene counts
    celltype_df : pandas DataFrame
        DataFrame containing cell type information
    location_df : pandas DataFrame
        DataFrame containing location information
    env_values_df : pandas DataFrame
        DataFrame containing environment values
    genename : str
        Name of the gene to analyze
    celltypename : str
        Name of the cell type to analyze
    envname : str
        Name of the environment variable to analyze
    axes : tuple or list of matplotlib axes, optional
        Axes to draw the plots on. If None, new figures are created.
        Use None in the tuple for plots you don't want to draw on your axes.
        Example: (None, ax1, ax2) will skip the first plot and use ax1 and ax2 for the others.
    plots_to_generate : tuple of bool, optional
        Tuple indicating which plots to generate (celltype PDP, env PDP, interaction/stratified)
        Default is (True, True, True) which generates all plots
    is_binary_celltype : bool, optional
        If True, treats cell type as binary indicator and creates a stratified PDP
        for the third plot instead of the 2D interaction. Default is False.
    n_neighbors : array-like, optional
        Range of neighbors to test for KNN regression. Default is np.arange(1, 21, 2).
        
    Returns:
    --------
    fig : matplotlib Figure or None
        The figure object if axes is None, otherwise None
    plots : list
        List of PartialDependenceDisplay objects or other plot objects
    """
    # Prepare the data
    y = normalized_counts_df[genename].to_numpy()
    y = y - np.mean(y)
    
    merged_df = pd.merge(celltype_df, location_df, left_index=True, right_index=True)
    merged_df['env'] = env_values_df[envname].tolist()
    merged_df['y'] = y.tolist()
    
    X = merged_df.drop(columns=['y'])
    y = merged_df['y']
    
    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('knn', KNeighborsRegressor())
    ])
    
    # Find optimal number of neighbors
    n_neighbors_range = n_neighbors 
    cv_scores = []
    if len(n_neighbors_range) == 0:
        raise ValueError("n_neighbors_range must not be empty")
    if len(n_neighbors_range) == 1:
        # If only one value is provided, use it directly
        optimal_n_neighbors = n_neighbors_range[0]
        print(f'Using provided n_neighbors: {optimal_n_neighbors}')
    else:
        for n in n_neighbors_range:
            pipeline.set_params(knn__n_neighbors=n)
            scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_scores.append(scores.mean())
        
        optimal_n_neighbors = n_neighbors_range[np.argmax(cv_scores)]
        print(f'Optimal number of neighbors: {optimal_n_neighbors}')
    
    pipeline.set_params(knn__n_neighbors=optimal_n_neighbors)
    pipeline.fit(X_train, y_train)
    
    test_score = pipeline.score(X_test, y_test)
    print(f'Test score (R^2): {test_score}')
    
    # Fit the model on all data for PDP
    pipeline.fit(X, y)
    
    # Determine if we need to create a figure or use provided axes
    num_plots = sum(plots_to_generate)
    need_new_figure = axes is None
    
    if need_new_figure:
        fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
        if num_plots == 1:
            axs = [axs]  # Make sure axs is always a list or array
    else:
        axs = axes
        if not isinstance(axs, (list, tuple, np.ndarray)):
            axs = [axs]
    
    # List to store plot objects
    plots = []
    plot_idx = 0
    
    # Plot 1: Celltype PDP
    if plots_to_generate[0]:
        curr_ax = None if (plot_idx >= len(axs) or axs[plot_idx] is None) else axs[plot_idx]
        
        if curr_ax is not None:
            pdp_display = PartialDependenceDisplay.from_estimator(
                pipeline, X, [celltypename], 
                grid_resolution=100, kind='both', ax=curr_ax
            )
                      
            curr_ax.set_title(f'Partial Dependence Plot for {genename} in {celltypename}', fontsize=title_fontsize)
            curr_ax.set_ylabel('Partial Dependence', fontsize=label_fontsize)
            curr_ax.set_xlabel(f'{celltypename} Feature Value', fontsize=label_fontsize)
            curr_ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

            plots.append(pdp_display)
        plot_idx += 1
    
    # Plot 2: Environment PDP
    if plots_to_generate[1]:
        curr_ax = None if (plot_idx >= len(axs) or axs[plot_idx] is None) else axs[plot_idx]
        
        if curr_ax is not None:
            pdp_display = PartialDependenceDisplay.from_estimator(
                pipeline, X, ['env'], 
                grid_resolution=100, kind='both', ax=curr_ax
            )
            
            curr_ax.set_title(f'Partial Dependence Plot for {genename} and {envname}', fontsize=title_fontsize)
            curr_ax.set_ylabel('Partial Dependence', fontsize=label_fontsize)
            curr_ax.set_xlabel(f'{envname} Feature Value', fontsize=label_fontsize)
            curr_ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            
            plots.append(pdp_display)
        plot_idx += 1
    
    # Plot 3: Either 2D interaction or stratified PDP
    if plots_to_generate[2]:
        curr_ax = None if (plot_idx >= len(axs) or axs[plot_idx] is None) else axs[plot_idx]
        
        if curr_ax is not None:
            if is_binary_celltype:
                # Optimized stratified PDP calculation
                unique_values = sorted(X[celltypename].unique())
                if len(unique_values) != 2:
                    print(f"Warning: Expected binary indicator for {celltypename} but found {len(unique_values)} unique values.")
                
                # Create the environment grid
                env_min, env_max = X['env'].min(), X['env'].max()
                env_grid = np.linspace(env_min, env_max, 100)
                
                # Calculate PDPs for each unique value of the celltype
                stratified_plots = []
                
                # Get feature names to properly index features
                feature_names = X.columns.tolist()
                env_idx = feature_names.index('env')
                line_data_to_return = [] if return_line_data else None

                for value in unique_values:
                    # Get indices where the celltype has this value
                    indices = np.where(X[celltypename] == value)[0]
                    
                    # Use scikit-learn's partial_dependence for vectorized calculation
                    pd_result = partial_dependence(
                        pipeline, X.iloc[indices], 
                        features=['env'], 
                        kind='average',
                        grid_resolution=100
                    )
                    if return_line_data:
                        line_data_to_return.append({
                            'label': f'{celltypename}={value}',
                            'x_values': pd_result["grid_values"][0],
                            'y_values': pd_result["average"][0]
                        })
                    
                    # Plot the result
                    line, = curr_ax.plot(pd_result["grid_values"][0], pd_result["average"][0], 
                                        label=f'{celltypename}={value}',
                                        linewidth=linewidth)
                    stratified_plots.append(line)
                
                    curr_ax.set_title(f'Stratified PDP for {genename} by {envname}\nand {celltypename}', fontsize=title_fontsize)
                    curr_ax.set_ylabel('Normalized Gene Expression', fontsize=label_fontsize)
                    curr_ax.set_xlabel(f'{envname} Feature Value', fontsize=label_fontsize)
                    curr_ax.legend(fontsize=label_fontsize, loc=legend_loc)
                    curr_ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                
                plots.append(stratified_plots)
            else:
                # Regular 2D interaction plot
                pdp_display = PartialDependenceDisplay.from_estimator(
                    pipeline, X, [('env', celltypename)],
                    grid_resolution=50, kind='average', ax=curr_ax
                )
                
                curr_ax.set_title(f'2D Interaction for {celltypename} and {envname}\nfor {genename}', fontsize=title_fontsize)
                curr_ax.set_xlabel(curr_ax.get_xlabel(), fontsize=label_fontsize)
                curr_ax.set_ylabel(curr_ax.get_ylabel(), fontsize=label_fontsize)
                curr_ax.tick_params(axis='both', which='both', length=0.001, labelsize=tick_fontsize)

                plots.append(pdp_display)
    
    if return_line_data and is_binary_celltype:
        if need_new_figure:
            plt.tight_layout()
            plt.show()
            return fig, plots, line_data_to_return
        else:
            return None, plots, line_data_to_return
    else:
        if need_new_figure:
            plt.tight_layout()
            plt.show()
            return fig, plots
        else:
            return None, plots
    
    
    

def adjust_ylim_for_legend(ax, legend, margin=1.1):
    """
    Adjusts the Y-axis limits of an axes object to make room for a legend.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object containing the plot and legend.
    legend : matplotlib.legend.Legend
        The legend object.
    margin : float, optional
        A multiplier to add a little extra padding. 1.1 means 10% padding.
    """
    # We need to draw the canvas to give the legend a definitive size and position
    fig = ax.get_figure()
    fig.canvas.draw()

    # Get the bounding box of the legend in pixels
    legend_bbox = legend.get_window_extent()
    
    # Get the bounding box of the axes in pixels
    ax_bbox = ax.get_window_extent()

    # Get the current y-limits in data coordinates
    ymin, ymax = ax.get_ylim()
    data_range = ymax - ymin

    # Determine how much of the axis height the legend occupies
    # This is the ratio of legend height in pixels to axis height in pixels
    fractional_height = legend_bbox.height / ax_bbox.height
    
    # Calculate the space needed in data coordinates
    space_needed = data_range * fractional_height * margin

    # Check the legend location to decide whether to adjust the top or bottom limit
    loc = legend.get_loc()
    
    if 'upper' in loc or 'top' in loc:
        ax.set_ylim(ymin, ymax + space_needed)
    elif 'lower' in loc or 'bottom' in loc:
        ax.set_ylim(ymin - space_needed, ymax)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

def create_enhanced_spotlight_pdp_plot(normalized_counts_df, celltype_df, location_df, env_values_df, 
                                      genename, celltypename, envname, 
                                      axes=None, n_neighbors=np.arange(1, 50, 4),
                                      cross_sections=[0, 0.5, 1], colors=None, 
                                      grid_resolution=50,                  
                                      linewidth=2,
                                      title_fontsize=14,
                                      label_fontsize=12,
                                      tick_fontsize=10,
                                      legend_loc='best',
                                      return_line_data=False):
    """
    Enhanced spotlight PDP with 2D interaction (left) and cross-sections at different celltype values (right).
    
    Parameters:
    -----------
    env_cross_sections : list, optional
        Celltype proportion values for cross-sections. Default is [0, 0.5, 1].
        These represent different celltype proportions to show as separate lines.
    """
    # Data preparation
    y = normalized_counts_df[genename].to_numpy()
    y = y - np.mean(y)
    
    merged_df = pd.merge(celltype_df, location_df, left_index=True, right_index=True)
    merged_df['env'] = env_values_df[envname].to_list()
    merged_df['y'] = y.tolist()
    
    X = merged_df.drop(columns=['y'])
    y = merged_df['y']
    
    # Model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('knn', KNeighborsRegressor())
    ])
    
    if len(n_neighbors) == 1:
        optimal_n_neighbors = n_neighbors[0]
        print(f'Using provided n_neighbors: {optimal_n_neighbors}')
    else:
        cv_scores = []
        for n in n_neighbors:
            pipeline.set_params(knn__n_neighbors=n)
            scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_scores.append(scores.mean())
        
        optimal_n_neighbors = n_neighbors[np.argmax(cv_scores)]
        print(f'Optimal number of neighbors: {optimal_n_neighbors}')
    
    pipeline.set_params(knn__n_neighbors=optimal_n_neighbors)
    pipeline.fit(X, y)
    
    # Create figure with each subplot being (6, 6)
    if axes is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        need_new_figure = True
    else:
        ax1, ax2 = axes
        fig = None
        need_new_figure = False
    
    plots = []
    
    # Left plot: 2D interaction plot (env × celltype)
    pdp_display = PartialDependenceDisplay.from_estimator(
        pipeline, X, [('env', celltypename)],
        grid_resolution=grid_resolution, kind='average', ax=ax1
    )
    
    ax1.set_title(f'2D Interaction: {envname} × {celltypename}\nfor {genename}', fontsize=title_fontsize)
    # env_min, env_max = X['env'].min(), X['env'].max()
    # celltype_min, celltype_max = X[celltypename].min(), X[celltypename].max()
    # ax1.set_xlim(env_min, env_max)
    # ax1.set_ylim(celltype_min, celltype_max)
    ax1.set_xlabel(f'{envname}', fontsize=label_fontsize)
    ax1.set_ylabel(f'{celltypename}', fontsize=label_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    plots.append(pdp_display)
    
    # Right plot: Cross-sections at different CELLTYPE proportions
    # X-axis: env density, Y-axis: gene expression, different lines for celltype proportions
    
    if colors is None:
        colors = ['#004488', '#BB5566', '#DDAA33']  # Default colors
    
    stratified_lines = []
    line_data_to_return = [] if return_line_data else None
    
    for i, celltype_val in enumerate(cross_sections):  # These are actually celltype values
        # Create synthetic dataset with fixed celltype proportion
        X_synthetic = X.copy()
        X_synthetic[celltypename] = celltype_val
        
        # Calculate partial dependence for the env feature (varying env, fixed celltype)
        pd_result = partial_dependence(
            pipeline, X_synthetic, 
            features=['env'],  # Vary environment density
            kind='average',
            grid_resolution=100
        )
        
        color = colors[i] if i < len(colors) else colors[i % len(colors)]
        label = f'{celltypename}={celltype_val}'
        if return_line_data:
            line_data_to_return.append({
                'label': f'{celltypename}={celltype_val}',
                'x_values': pd_result["grid_values"][0],
                'y_values': pd_result["average"][0]
            })
        
        line, = ax2.plot(pd_result["grid_values"][0], pd_result["average"][0], 
                        color=color, label=label, linewidth=linewidth)
        stratified_lines.append(line)
    
    ax2.set_title(f'Cross-sections at Different {celltypename} Values\nfor {genename}', fontsize=title_fontsize)
    ax2.set_xlabel(f'{envname}', fontsize=label_fontsize)
    ax2.set_ylabel('Normalized Gene Expression', fontsize=label_fontsize)
    ax2.legend(fontsize=label_fontsize, loc=legend_loc)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    plots.append(stratified_lines)
    
    if return_line_data:
        if need_new_figure:
            plt.tight_layout()
            plt.show()
            return fig, plots, line_data_to_return
        else:
            return None, plots, line_data_to_return
    else:
        if need_new_figure:
            plt.tight_layout()
            plt.show()
            return fig, plots
        else:
            return None, plots


import pandas as pd
from matplotlib_venn import venn3, venn2
import matplotlib.pyplot as plt
import math
import numpy as np

def get_significant_genes_from_df(df, celltype, fdr_threshold=0.05, fdr_method=''):
    """
    Extract significant genes for a specific celltype from a DataFrame
    
    Parameters:
    - df: DataFrame containing results for one method
    - celltype: cell type to filter for
    - fdr_threshold: significance threshold
    
    Returns:
    - set of significant gene names
    """
    # Filter for cell type
    df_filtered = df[df['Celltype'] == celltype]

    if fdr_method:
        # Determine which p-value column to use
        if (fdr_method in df_filtered.columns and 
            not df_filtered[fdr_method].isna().all()):
            significant = df_filtered[df_filtered[fdr_method] <= fdr_threshold]
        else:
            raise ValueError(f"Specified FDR method '{fdr_method}' not found or does not contain any values in DataFrame")

    else:
        if ('FDR_permutation_adjusted_p_value' in df_filtered.columns and 
            not df_filtered['FDR_permutation_adjusted_p_value'].isna().all()):
            significant = df_filtered[df_filtered['FDR_permutation_adjusted_p_value'] <= fdr_threshold]
        else:
            if 'FDR_p_value' in df_filtered.columns and not df_filtered['FDR_p_value'].isna().all():
                significant = df_filtered[df_filtered['FDR_p_value'] <= fdr_threshold]
            else:
                raise ValueError("No FDR p-value columns found in DataFrame")
    
    return set(significant['Gene'])

def create_venn_from_dataframes(results_dfs, method_names, celltypes, 
                               nrows=None, ncols=None, figsize=None, 
                               fdr_threshold=0.05, output_file=None):
    """
    Create Venn diagrams comparing methods for each cell type using DataFrames
    
    Parameters:
    - results_dfs: list of DataFrames, each containing results for one method
    - method_names: display names for each method (corresponding to results_dfs)
    - celltypes: list of cell types to analyze
    - nrows, ncols: grid layout dimensions
    - figsize: figure size (width, height) in inches
    - fdr_threshold: significance threshold
    - output_file: path to save figure (if None, just displays)
    
    Returns:
    - fig, axes: figure and axes objects
    """
    # Validate inputs
    if not isinstance(results_dfs, list):
        results_dfs = [results_dfs]  # Convert single DataFrame to list
        
    if len(results_dfs) != len(method_names):
        raise ValueError("Number of DataFrames must match number of method names")
    
    # Check for required columns in all DataFrames
    required_cols = ['Gene', 'Celltype']
    for i, df in enumerate(results_dfs):
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame for {method_names[i]} is missing columns: {missing_cols}")
            
        # At least one FDR column should be present
        if 'FDR_p_value' not in df.columns and 'FDR_permutation_adjusted_p_value' not in df.columns:
            raise ValueError(f"DataFrame for {method_names[i]} has no FDR p-value columns")
    
    n_plots = len(celltypes)
    
    # Calculate layout if not fully specified
    if nrows is None and ncols is None:
        nrows, ncols = 1, n_plots
    elif nrows is None:
        nrows = math.ceil(n_plots / ncols)
    elif ncols is None:
        ncols = math.ceil(n_plots / nrows)
    
    # Calculate default figsize if not provided
    if figsize is None:
        figsize = (6 * ncols, 6 * nrows)
    
    # Create figure and axes grid
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Convert axes to 2D array if it's 1D or single axis
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Process each celltype
    for idx, celltype in enumerate(celltypes):
        if idx >= nrows * ncols:
            print(f"Warning: Not enough space to plot {celltype}")
            break
            
        row_idx = idx // ncols
        col_idx = idx % ncols
        
        print(f"\nProcessing celltype: {celltype}")
        
        # Get gene sets for each method/DataFrame
        gene_sets = []
        for df in results_dfs:
            if celltype in df['Celltype'].unique():
                gene_set = get_significant_genes_from_df(df, celltype, fdr_threshold)
                gene_sets.append(gene_set)
            else:
                print(f"Warning: Celltype '{celltype}' not found in DataFrame for method '{method_names[len(gene_sets)]}'")
                gene_sets.append(set())  # Empty set if celltype not found
        
        # Print statistics
        print("\nNumber of significant genes in each method:")
        for name, genes in zip(method_names, gene_sets):
            print(f"{name}: {len(genes)}")
        
        if len(gene_sets) == 3:
            intersection_all = gene_sets[0] & gene_sets[1] & gene_sets[2]
            print(f"\nGenes in all three methods: {len(intersection_all)}")
            print(f"Genes in {method_names[0]} & {method_names[1]}: {len(gene_sets[0] & gene_sets[1])}")
            print(f"Genes in {method_names[0]} & {method_names[2]}: {len(gene_sets[0] & gene_sets[2])}")
            print(f"Genes in {method_names[1]} & {method_names[2]}: {len(gene_sets[1] & gene_sets[2])}")
            
            # Optionally print the actual genes in the intersection
            if len(intersection_all) > 0:
                print(f"Genes found in all methods: {sorted(intersection_all)}")
        
        elif len(gene_sets) == 2:
            intersection = gene_sets[0] & gene_sets[1]
            print(f"\nGenes in both methods: {len(intersection)}")
            if len(intersection) > 0:
                print(f"Genes found in both methods: {sorted(intersection)}")
        
        # Set the current subplot
        plt.sca(axes[row_idx, col_idx])
        
        # Create Venn diagram
        if len(gene_sets) == 2:
            venn2(gene_sets, set_labels=method_names)
        elif len(gene_sets) == 3:
            venn3(gene_sets, set_labels=method_names)
        else:
            plt.text(0.5, 0.5, f"Cannot create Venn diagram for {len(gene_sets)} sets",
                     ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
        
        plt.title(f'{celltype}\n(FDR ≤ {fdr_threshold})')
    
    # Remove empty subplots if any
    for idx in range(len(celltypes), nrows * ncols):
        row_idx = idx // ncols
        col_idx = idx % ncols
        fig.delaxes(axes[row_idx, col_idx])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_file}")
    
    plt.show()
    
    return fig, axes


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_p_value_histogram(true_values, permutation_values, title='', 
                             bin_width=0.01, legend_position=(0.55, 0.8), 
                             xlim=None, ylim=None, show_grid=False, ax=None,
                             show_inflation=True):  # Added new parameter
    """
    Create a histogram comparing true p-values against permutation p-values.
    
    Parameters:
    -----------
    true_values : array-like
        Array or list of true p-values
    permutation_values : array-like
        Array or list of permutation p-values
    title : str, optional
        Plot title
    bin_width : float, optional
        Width of histogram bins
    legend_position : tuple, optional
        Position of the legend (x, y) relative to the plot
    xlim : tuple, optional
        (min, max) limits for the x-axis. If None, auto-scaling is used.
    ylim : tuple, optional
        (min, max) limits for the y-axis. If None, auto-scaling is used.
    show_grid : bool, optional
        Whether to display grid lines (default: False)
    ax : matplotlib axis, optional
        Axis to plot on. If None, a new figure is created
    show_inflation : bool, optional
        Whether to display the genomic inflation factor (default: True)
        
    Returns:
    --------
    ax : matplotlib axes object
    """
    # Create DataFrames for each set of values
    true_df = pd.DataFrame({'p_value': true_values, 'type': 'True'})
    perm_df = pd.DataFrame({'p_value': permutation_values, 'type': 'Permutation'})
    data = pd.concat([true_df, perm_df], ignore_index=True)
    
    # Create a new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set the style - use ticks instead of whitegrid to have more control
    sns.set_style("ticks")
    
    # Plot the histograms using different colors
    sns.histplot(data=data, x='p_value', hue='type', binwidth=bin_width, 
                 alpha=0.5, element='bars', common_norm=False, ax=ax,
                 palette={'Permutation': '#66c2a5', 'True': 'navy'}, 
                 edgecolor='white')
    
    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel('P values')
    ax.set_ylabel('Count')
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Configure grid visibility
    if show_grid:
        ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.7, alpha=0.7)
    else:
        ax.grid(False)  # Turn off all grid lines
    
    # Always show the border box
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    # Customize legend with transparent background
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, title='', loc='upper left', 
                      bbox_to_anchor=legend_position)
    legend.get_frame().set_facecolor('none')  # Make background transparent
    legend.get_frame().set_linewidth(0)  # Remove the legend border
    
    # Calculate and display genomic inflation factor if requested
    if show_inflation:
        # Convert permutation p-values to chi-square statistics
        import numpy as np
        from scipy import stats
        
        # Filter out any extreme values (0 or 1) that would cause problems in the calculation
        valid_perm_pvals = np.array(permutation_values)
        valid_perm_pvals = valid_perm_pvals[~np.isnan(valid_perm_pvals)]
        valid_perm_pvals = np.clip(valid_perm_pvals, 1e-16, 1-1e-16)  # Avoid extreme values
        
        # Calculate chi-square statistics from p-values (assuming 1 degree of freedom)
        chi2_values = stats.chi2.ppf(1 - valid_perm_pvals, df=1)
        
        # Calculate genomic inflation factor (lambda)
        # Lambda is the ratio of median chi-square to its expected value (0.455 for 1 df)
        expected_median = 0.455  # Expected median of chi-square with 1 df
        inflation_factor = np.median(chi2_values) / expected_median
        
        # Add the inflation factor text to the top right of the plot
        # Use transform=ax.transAxes to use relative coordinates (0-1)
        ax.text(0.95, 0.95, f'λ = {inflation_factor:.3f}', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                ha='right', va='top', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    
    return ax

def create_p_value_histogram_grid(true_values_list, permutation_values_list, titles, 
                       figsize=(15, 12), rows=None, cols=None, 
                       bin_width=0.01, xlim=None, ylim=None,
                       show_grid=False, show_plot=True):
    """
    Create a grid of p-value histograms
    
    Parameters:
    -----------
    true_values_list : list of arrays
        List containing arrays of true p-values for each subplot
    permutation_values_list : list of arrays
        List containing arrays of permutation p-values for each subplot
    titles : list of str
        List of titles for each subplot
    figsize : tuple, optional
        Figure size
    bin_width : float, optional
        Width of histogram bins
    xlim : tuple or list of tuples, optional
        Either a single (min, max) tuple for all plots, or a list of tuples for each plot
    ylim : tuple or list of tuples, optional
        Either a single (min, max) tuple for all plots, or a list of tuples for each plot
    show_grid : bool, optional
        Whether to display grid lines (default: False)
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    # Calculate grid dimensions
    n_plots = len(true_values_list)
    if rows is None and cols is None:
        # If neither is specified, calculate a square grid
        rows = int(np.ceil(np.sqrt(n_plots)))
        cols = int(np.ceil(n_plots / rows))
    elif rows is None:
        # If only cols is specified, calculate rows
        rows = int(np.ceil(n_plots / cols))
    elif cols is None:
        # If only rows is specified, calculate cols
        cols = int(np.ceil(n_plots / rows))
    
    # Create subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten axes array for easy iteration
    if rows > 1 and cols > 1:
        axes_flat = axes.flatten()
    elif rows == 1 and cols > 1:
        axes_flat = axes
    elif rows > 1 and cols == 1: 
        axes_flat = axes
    else:
        axes_flat = [axes]
    
    # Process xlim and ylim
    if xlim is not None and not isinstance(xlim[0], tuple):
        # If a single tuple is provided, use it for all plots
        xlim_list = [xlim] * n_plots
    else:
        # Otherwise, it should be a list of tuples
        xlim_list = xlim if xlim is not None else [None] * n_plots
        
    if ylim is not None and not isinstance(ylim[0], tuple):
        # If a single tuple is provided, use it for all plots
        ylim_list = [ylim] * n_plots
    else:
        # Otherwise, it should be a list of tuples
        ylim_list = ylim if ylim is not None else [None] * n_plots
    
    # Create each subplot
    for i in range(n_plots):
        create_p_value_histogram(
            true_values_list[i], 
            permutation_values_list[i],
            titles[i],
            bin_width=bin_width,
            xlim=xlim_list[i],
            ylim=ylim_list[i],
            show_grid=show_grid,
            ax=axes_flat[i]
        )
    
    # Hide unused subplots
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()

    if show_plot:
        plt.show()

    return fig, axes


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib import cm

def plot_spatial_gene_env_correlation(
    normalized_counts_df, celltype_df, location_df, env_values_df,
    genename, celltypename, envname,
    point_size=5, transparency=0.3, 
    contour_levels=None,
    colormap='viridis',
    highlight_colors=None,
    figsize=(12, 10),
    ax=None,
    dpi=100,
    interpolation_method='cubic',
    interpolation_function=None,
    grid_resolution=100,
    contour_alpha=0.5,
    show_colorbar=True,
    highlight_point_scale=1.5,
    background_points=True,
    exp_cell_ratio=True
):
    """
    Plot cells in spatial context with gene expression and environmental variable correlation.
    
    Parameters:
    -----------
    normalized_counts_df : pandas DataFrame
        DataFrame containing normalized gene counts
    celltype_df : pandas DataFrame
        DataFrame containing cell type information (should have a column that matches celltypename)
    location_df : pandas DataFrame
        DataFrame containing x,y coordinates (should have 'x' and 'y' columns)
    env_values_df : pandas DataFrame
        DataFrame containing environment values
    genename : str
        Name of the gene to analyze
    celltypename : str
        Name of the cell type column to analyze
    envname : str
        Name of the environment variable to analyze
    point_size : int, optional
        Size of the points in the scatter plot
    transparency : float, optional
        Alpha value for transparency of background cells (0-1)
    contour_levels : list, optional
        Specific contour levels to use. If None, auto-generated
    colormap : str, optional
        Colormap to use for contour plot
    highlight_colors : list, optional
        Three colors to use for highlighting cells by gene expression level
    figsize : tuple, optional
        Figure size (width, height) in inches
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created
    dpi : int, optional
        DPI for the figure
    interpolation_method : str, optional
        Method for scipy.interpolate.griddata: 'nearest', 'linear', or 'cubic'
    interpolation_function : callable, optional
        Custom function to calculate env values at arbitrary points: 
        f(x_points, y_points) -> env_values
    grid_resolution : int, optional
        Resolution of the grid for interpolation
    contour_alpha : float, optional
        Alpha transparency for contour plot
    show_colorbar : bool, optional
        Whether to show the colorbar
    highlight_point_scale : float, optional
        Scale factor for highlighted points compared to background points
    background_points : bool, optional
        Whether to show background (non-target cell type) points
        
    Returns:
    --------
    fig : matplotlib Figure
        The created figure (or None if ax was provided)
    ax : matplotlib Axes
        The axes with the plot
    """
    # Example usage with default interpolation:
    # fig, ax = plot_spatial_gene_env_correlation(
    #     normalized_counts_df, celltype_df, location_df, env_values_df,
    #     genename='YourGene', celltypename='YourCellType', envname='YourEnvVariable',
    #     point_size=8, grid_resolution=150, colormap='plasma'
    # )
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = None
    
    # Prepare the data
    merged_df = pd.DataFrame(index=normalized_counts_df.index)
    merged_df['x'] = location_df['x']
    merged_df['y'] = location_df['y']
    merged_df['gene_expression'] = normalized_counts_df[genename]
    merged_df['celltype'] = celltype_df.idxmax(axis=1) # only get the cell type with the highest value
    merged_df['env'] = env_values_df[envname]

    if exp_cell_ratio:
        # Calculate the ratio of environment value to cell type value
        merged_df['gene_expression'] = merged_df['gene_expression'] / celltype_df[celltypename]
    
    # Extract coordinates
    x = merged_df['x'].values
    y = merged_df['y'].values
    
    # 1. Plot all cells as gray semi-transparent dots
    if background_points:
        ax.scatter(x, y, s=point_size, color='gray', alpha=transparency, label='Other cells')
    
    # 2. Create contour plot for environmental variable
    # Generate a regular grid for interpolation
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    grid_x, grid_y = np.linspace(x_min, x_max, grid_resolution), np.linspace(y_min, y_max, grid_resolution)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    
    # Get environmental values
    env_values = merged_df['env'].values
    
    # Interpolate the environmental values onto the grid
    if interpolation_function is None:
        grid_env = griddata((x, y), env_values, (grid_x, grid_y), method=interpolation_method)
    else:
        # Use custom interpolation function
        grid_points_x = grid_x.flatten()
        grid_points_y = grid_y.flatten()
        grid_env_flat = interpolation_function(grid_points_x, grid_points_y)
        grid_env = grid_env_flat.reshape(grid_x.shape)
    
    # Generate contour levels if not provided
    if contour_levels is None:
        contour_levels = np.linspace(env_values.min(), env_values.max(), 11)
    
    # Create contour plot
    contour = ax.contourf(grid_x, grid_y, grid_env, levels=contour_levels, 
                         cmap=colormap, alpha=contour_alpha)
    
    # Add contour lines for clarity
    contour_lines = ax.contour(grid_x, grid_y, grid_env, levels=contour_levels, 
                              colors='black', alpha=0.7, linewidths=0.7)
    ax.clabel(contour_lines, fmt='%2.1f', fontsize=8, inline=True)

    # Add colorbar for the contour plot
    if show_colorbar:
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(f'{envname} Value')
    
    # 3. Highlight cells of specific type by gene expression level
    cells_of_type = merged_df[merged_df['celltype'] == celltypename]
    
    if len(cells_of_type) == 0:
        print(f"Warning: No cells found with type '{celltypename}'")
        print(f"Available cell types: {merged_df['celltype'].unique()}")
    else:
        # Sort by gene expression
        cells_of_type = cells_of_type.sort_values('gene_expression')
        
        # Split into three equal groups
        n = len(cells_of_type)
        first_third = cells_of_type.iloc[:n//3]
        middle_third = cells_of_type.iloc[n//3:2*n//3]
        last_third = cells_of_type.iloc[2*n//3:]
        
        # Default highlight colors if not provided
        if highlight_colors is None:
            highlight_colors = ['blue', 'purple', 'red']
        
        # Calculate highlighted point size
        highlight_size = point_size * highlight_point_scale
        
        # Plot each group with different colors
        ax.scatter(first_third['x'], first_third['y'], s=highlight_size, 
                  color=highlight_colors[0], label=f'{celltypename}: Low {genename}')
        ax.scatter(middle_third['x'], middle_third['y'], s=highlight_size, 
                  color=highlight_colors[1], label=f'{celltypename}: Medium {genename}')
        ax.scatter(last_third['x'], last_third['y'], s=highlight_size, 
                  color=highlight_colors[2], label=f'{celltypename}: High {genename}')
    
    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Gene Expression of {genename} in {celltypename} Cells\n'
                f'Correlated with {envname} Environment')
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Validate interpolation if using built-in methods
    if interpolation_function is None:
        grid_env_at_cells = griddata((x, y), env_values, (x, y), method=interpolation_method)
        mean_abs_error = np.mean(np.abs(grid_env_at_cells - env_values))
        print(f"Interpolation validation - Mean absolute error: {mean_abs_error:.4f}")
    
    if fig is not None:
        plt.tight_layout()
    
    return fig, ax

# Example with custom interpolation function
def example_custom_interpolation(normalized_counts_df, celltype_df, location_df, env_values_df,
                              genename, celltypename, envname):
    """Example showing how to use a custom interpolation function"""
    
    # Example custom interpolation function (e.g., inverse distance weighting)
    def custom_idw_interpolation(x_targets, y_targets, power=2):
        """
        Inverse distance weighting interpolation
        
        Parameters:
        -----------
        x_targets, y_targets : array-like
            Coordinates where to interpolate
        power : float
            Power parameter controlling how quickly influence drops with distance
            
        Returns:
        --------
        env_interpolated : array-like
            Interpolated environmental values at target points
        """

        # Get source points and values from the dataset
        x_source = location_df['x'].values
        y_source = location_df['y'].values
        z_source = env_values_df[envname].values
        
        # Initialize output array
        env_interpolated = np.zeros_like(x_targets, dtype=float)
        
        # For each target point
        for i in range(len(x_targets)):
            # Calculate distances to all source points
            distances = np.sqrt((x_targets[i] - x_source)**2 + (y_targets[i] - y_source)**2)
            
            # Handle the case where a target point exactly matches a source point
            if np.any(distances == 0):
                idx = np.where(distances == 0)[0][0]
                env_interpolated[i] = z_source[idx]
            else:
                # Calculate weights as inverse of distance raised to power
                weights = 1.0 / (distances ** power)
                
                # Normalize weights to sum to 1
                weights = weights / np.sum(weights)
                
                # Calculate weighted average
                env_interpolated[i] = np.sum(weights * z_source)
        
        return env_interpolated
    
    # Create the plot with custom interpolation
    fig, ax = plot_spatial_gene_env_correlation(
        normalized_counts_df, celltype_df, location_df, env_values_df,
        genename, celltypename, envname,
        interpolation_function=custom_idw_interpolation,
        colormap='plasma',
        contour_levels=20,
        highlight_colors=['darkblue', 'darkgreen', 'darkred']
    )
    
    return fig, ax


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors as mcolors

def plot_spatial_gene_expression_contour(
    normalized_counts_df, celltype_df, location_df, env_values_df,
    genename, celltypename, envname,
    point_size=5, transparency=0.3, 
    contour_levels=None,
    colormap='viridis',
    highlight_colors=None,
    figsize=(12, 10),
    ax=None,
    dpi=100,
    interpolation_method='cubic',
    interpolation_function=None,
    grid_resolution=100,
    contour_alpha=0.5,
    show_colorbar=True,
    highlight_point_scale=1.5,
    background_points=True,
    lines_fontsize=8,
    legend_loc='lower right',
    equal_aspect=True
):
    """
    Plot cells in spatial context with gene expression and environmental variable correlation.
    
    Parameters:
    -----------
    normalized_counts_df : pandas DataFrame
        DataFrame containing normalized gene counts
    celltype_df : pandas DataFrame
        DataFrame containing cell type information (should have a column that matches celltypename)
    location_df : pandas DataFrame
        DataFrame containing x,y coordinates (should have 'x' and 'y' columns)
    env_values_df : pandas DataFrame
        DataFrame containing environment values
    genename : str
        Name of the gene to analyze
    celltypename : str
        Name of the cell type column to analyze
    envname : str
        Name of the environment variable to analyze
    point_size : int, optional
        Size of the points in the scatter plot
    transparency : float, optional
        Alpha value for transparency of background cells (0-1)
    contour_levels : list, optional
        Specific contour levels to use. If None, auto-generated
    colormap : str, optional
        Colormap to use for contour plot
    highlight_colors : list, optional
        Three colors to use for highlighting cells by gene expression level
    figsize : tuple, optional
        Figure size (width, height) in inches
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created
    dpi : int, optional
        DPI for the figure
    interpolation_method : str, optional
        Method for scipy.interpolate.griddata: 'nearest', 'linear', or 'cubic'
    interpolation_function : callable, optional
        Custom function to calculate env values at arbitrary points: 
        f(x_points, y_points) -> env_values
    grid_resolution : int, optional
        Resolution of the grid for interpolation
    contour_alpha : float, optional
        Alpha transparency for contour plot
    show_colorbar : bool, optional
        Whether to show the colorbar
    highlight_point_scale : float, optional
        Scale factor for highlighted points compared to background points
    background_points : bool, optional
        Whether to show background (non-target cell type) points
        
    Returns:
    --------
    fig : matplotlib Figure
        The created figure (or None if ax was provided)
    ax : matplotlib Axes
        The axes with the plot
    """
    # Example usage with default interpolation:
    # fig, ax = plot_spatial_gene_env_correlation(
    #     normalized_counts_df, celltype_df, location_df, env_values_df,
    #     genename='YourGene', celltypename='YourCellType', envname='YourEnvVariable',
    #     point_size=8, grid_resolution=150, colormap='plasma'
    # )
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = None
    
    # Prepare the data
    merged_df = pd.DataFrame(index=normalized_counts_df.index)
    merged_df['x'] = location_df['x']
    merged_df['y'] = location_df['y']
    merged_df['gene_expression'] = normalized_counts_df[genename]
    merged_df['celltype'] = celltype_df.idxmax(axis=1) # only get the cell type with the highest value
    merged_df['env'] = env_values_df[envname]
    
    # Extract coordinates
    x = merged_df['x'].values
    y = merged_df['y'].values
    
    # 1. Plot all cells as gray semi-transparent dots
    if background_points:
        ax.scatter(x, y, s=point_size, color='gray', alpha=transparency, label='Other cells')
    
    # 2. Create contour plot for environmental variable
    # Generate a regular grid for interpolation
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    grid_x, grid_y = np.linspace(x_min, x_max, grid_resolution), np.linspace(y_min, y_max, grid_resolution)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    
    # Get environmental values
    env_values = merged_df['env'].values
    
    # Interpolate the environmental values onto the grid
    if interpolation_function is None:
        grid_env = griddata((x, y), env_values, (grid_x, grid_y), method=interpolation_method)
    else:
        # Use custom interpolation function
        grid_points_x = grid_x.flatten()
        grid_points_y = grid_y.flatten()
        grid_env_flat = interpolation_function(grid_points_x, grid_points_y)
        grid_env = grid_env_flat.reshape(grid_x.shape)
    
    # Generate contour levels if not provided
    if contour_levels is None:
        contour_levels = np.linspace(env_values.min(), env_values.max(), 10)
    
    # Create contour plot
    contour = ax.contourf(grid_x, grid_y, grid_env, levels=contour_levels, 
                         cmap=colormap, alpha=contour_alpha)
    
        # in each env bin defined by the contour levels.
    cells_of_type = merged_df[merged_df['celltype'] == celltypename]
    avg_expr_list = []
    for i in range(len(contour_levels) - 1):
        lower = contour_levels[i]
        upper = contour_levels[i+1]
        bin_cells = cells_of_type[(cells_of_type['env'] >= lower) & (cells_of_type['env'] < upper)]
        if len(bin_cells) > 0:
            avg_expr = bin_cells['gene_expression'].mean()
        else:
            avg_expr = np.nan
        avg_expr_list.append(avg_expr)

    # Set up normalization for the average expressions in order to map to the colormap.
    valid_expr = [v for v in avg_expr_list if not np.isnan(v)]
    if valid_expr:
        norm = mcolors.Normalize(vmin=min(valid_expr), vmax=max(valid_expr))
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    
    # Override the fill-colors of each contour region with the color corresponding to the average gene expression.
    # Here we iterate over the contour collections (each corresponds to a region between two env boundaries).
    for i, collection in enumerate(contour.collections):
        # Get the average gene expression value for this contour bin
        avg_val = avg_expr_list[i] if i < len(avg_expr_list) else np.nan
        if np.isnan(avg_val):
            # If no data, assign a default (here transparent white)
            region_color = (1, 1, 1, 0.0)
        else:
            region_color = plt.get_cmap(colormap)(norm(avg_val))
        collection.set_facecolor(region_color)
    
    # Add contour lines for clarity
    contour_lines = ax.contour(grid_x, grid_y, grid_env, levels=contour_levels, 
                              colors='black', alpha=0.7, linewidths=0.7)
    ax.clabel(contour_lines, fmt='%2.1f', fontsize=lines_fontsize, inline=True)

    # Add colorbar for the contour plot
    if show_colorbar:
        from matplotlib.cm import ScalarMappable  # make sure this import is available at the top
        sm = ScalarMappable(cmap=colormap, norm=norm)  # norm is computed based on average gene expression values
        sm.set_array([])  # Dummy array for the mappable
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f'Average {genename} Expression in {celltypename}')
    
    # 3. Highlight cells of specific type by gene expression level
    
    if len(cells_of_type) == 0:
        print(f"Warning: No cells found with type '{celltypename}'")
        print(f"Available cell types: {merged_df['celltype'].unique()}")
    else:
        # Sort by gene expression
        cells_of_type = cells_of_type.sort_values('gene_expression')
        
        # Split into three equal groups
        n = len(cells_of_type)
        first_third = cells_of_type.iloc[:n//3]
        middle_third = cells_of_type.iloc[n//3:2*n//3]
        last_third = cells_of_type.iloc[2*n//3:]
        
        # Default highlight colors if not provided
        if highlight_colors is None:
            highlight_colors = ['blue', 'purple', 'red']
        
        # Calculate highlighted point size
        highlight_size = point_size * highlight_point_scale
        
        # Plot each group with different colors
        ax.scatter(first_third['x'], first_third['y'], s=highlight_size, 
                  color=highlight_colors[0], label=f'{celltypename}: Low {genename}')
        ax.scatter(middle_third['x'], middle_third['y'], s=highlight_size, 
                  color=highlight_colors[1], label=f'{celltypename}: Medium {genename}')
        ax.scatter(last_third['x'], last_third['y'], s=highlight_size, 
                  color=highlight_colors[2], label=f'{celltypename}: High {genename}')
    
    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Gene Expression of {genename} in {celltypename} Cells\n'
                f'Correlated with {envname} Environment')
    
    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')
    
    # Add legend
    if legend_loc is not None:
        ax.legend(loc=legend_loc)
    
    # Validate interpolation if using built-in methods
    if interpolation_function is None:
        grid_env_at_cells = griddata((x, y), env_values, (x, y), method=interpolation_method)
        mean_abs_error = np.mean(np.abs(grid_env_at_cells - env_values))
        print(f"Interpolation validation - Mean absolute error: {mean_abs_error:.4f}")
    
    if fig is not None:
        plt.tight_layout()
    
    return fig, ax


def plot_significant_counts(p_values_list, method_names=None, colors=None, 
                           fdr_thresholds=None, title="Significance Counts", 
                           figsize=None, ax=None, output_path=None, legend_loc='best'):
    """
    Create a barplot showing the number of significant values at different FDR thresholds
    
    Parameters:
    -----------
    p_values_list : list of arrays
        List containing arrays of p-values for each method
    method_names : list of str, optional
        Names of the methods. If not provided, defaults to "method_1", "method_2", etc.
    colors : list of str, optional
        Colors for each method. If not provided, default colors will be used.
    fdr_thresholds : list of float, optional
        FDR threshold values for x-axis. Default is 0.01 to 0.1 by 0.01.
    title : str, optional
        Title of the plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If not provided, new figure and axes will be created.
    output_path : str, optional
        Path to save the figure. If not provided, figure will not be saved.
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Validate input
    num_methods = len(p_values_list)
    
    # Set default method names if not provided
    if method_names is None:
        method_names = [f"method_{i+1}" for i in range(num_methods)]
    elif len(method_names) != num_methods:
        raise ValueError(f"Length of method_names ({len(method_names)}) must match number of p-value lists ({num_methods})")
    
    # Set default colors if not provided
    if colors is None:
        # Use a colormap to generate colors
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(num_methods)]
    elif len(colors) != num_methods:
        raise ValueError(f"Length of colors ({len(colors)}) must match number of p-value lists ({num_methods})")
    
    # Set default FDR thresholds if not provided
    if fdr_thresholds is None:
        fdr_thresholds = np.arange(0.01, 0.11, 0.01)
    
    # Create a new figure if ax is not provided
    create_new_fig = ax is None
    if create_new_fig:
        if figsize is None:
            figsize = (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Count significant p-values for each method at each threshold
    significant_counts = []
    for p_values in p_values_list:
        counts = []
        for threshold in fdr_thresholds:
            counts.append(sum(np.array(p_values) < threshold))
        significant_counts.append(counts)
    
    # Create DataFrame for plotting
    data = []
    for i, method in enumerate(method_names):
        for j, threshold in enumerate(fdr_thresholds):
            data.append({
                'FDR': f"{threshold:.2f}",
                'Count': significant_counts[i][j],
                'Method': method
            })
    df = pd.DataFrame(data)
    
    # Set up the x positions for the bars
    x = np.arange(len(fdr_thresholds))
    width = 0.8 / num_methods  # Width of bars with some spacing
    
    # Create the barplot
    for i, method in enumerate(method_names):
        method_data = df[df['Method'] == method]
        ax.bar(x + (i - num_methods/2 + 0.5) * width, 
               method_data['Count'], 
               width, 
               label=method,
               color=colors[i],
               edgecolor='black')
    
    # Add labels and styling
    ax.set_title(title)
    ax.set_xlabel('FDR')
    ax.set_ylabel('Number')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{threshold:.2f}" for threshold in fdr_thresholds], rotation=45)
    if isinstance(legend_loc, tuple) and len(legend_loc) == 3:
        ax.legend(loc=legend_loc[0], bbox_to_anchor=legend_loc[1:], ncol=legend_loc[2])
    else:
        ax.legend(loc=legend_loc)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure if output_path is provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show figure if it's not a subplot
    if create_new_fig and not plt.isinteractive():
        plt.show()
        
    return fig, ax


def plot_roc_grid_seeds_aggregate_CI(
    simulation_indexes, simulation_seeds, lmm_dir, cside_dir,
    true_signal_dir, output_dir, all_true_positives=True, aggregate='all', ci=0.95,
    celltypes_to_draw=None   # New argument
):
    """
    Plot ROC grid (2 columns) for specified cell types (if given) for each simulation index,
    aggregating by seeds (show all seed lines OR mean+empirical CI band)
    for CYGNET (LMM) and CSIDE only. 
    Only skip a simulation if neither method has any seed file.
    """
    os.makedirs(output_dir, exist_ok=True)

    methods = [
        {'name': 'CYGNET', 'dir': lmm_dir, 'color': 'blue'},
        {'name': 'CSIDE', 'dir': cside_dir, 'color': 'red'}
    ]
    alpha = (1 - ci) / 2

    for simulation_index in simulation_indexes:
        # ---- Check if at least one result file (for either method, any seed) exists ----
        found_any_file = False
        for simulation_seed in simulation_seeds:
            lmm_path = os.path.join(lmm_dir, f"sim_{simulation_index}_{simulation_seed}_lmm_fdr_all_results.csv")
            cside_path = os.path.join(cside_dir, f"sim_{simulation_index}_{simulation_seed}_cside_fdr_all_results.csv")
            if os.path.exists(lmm_path) or os.path.exists(cside_path):
                found_any_file = True
                break
        if not found_any_file:
            print(f"Skip sim {simulation_index}: no result files found for any method/seed")
            continue

        # Gather all cell types & gene-cell pairs
        all_cell_types = set()
        all_gene_celltype_pairs = set()
        true_positive_sets = {}

        for simulation_seed in simulation_seeds:
            true_signal_path = os.path.join(
                true_signal_dir, f"sim_{simulation_index}_{simulation_seed}_true_signals.csv"
            )
            if not os.path.exists(true_signal_path):
                continue
            true_signals = pd.read_csv(true_signal_path)
            true_positive_sets[simulation_seed] = set(zip(true_signals['gene_id'], true_signals['celltype']))
            all_cell_types.update(true_signals['celltype'].unique())

            for method in methods:
                if method['name'] == 'CYGNET':
                    file_path = os.path.join(method['dir'], f"sim_{simulation_index}_{simulation_seed}_lmm_fdr_all_results.csv")
                else:
                    file_path = os.path.join(method['dir'], f"sim_{simulation_index}_{simulation_seed}_cside_fdr_all_results.csv")
                if os.path.exists(file_path):
                    results = pd.read_csv(file_path)
                    gene_col = 'Gene' if 'Gene' in results.columns else 'gene_id'
                    celltype_col = 'Celltype' if 'Celltype' in results.columns else 'celltype'
                    all_cell_types.update(results[celltype_col].unique())
                    all_gene_celltype_pairs.update(zip(results[gene_col], results[celltype_col]))

        # === Only use the requested cell types if provided ===
        if celltypes_to_draw is not None:
            # Intersect with those actually present in data
            all_cell_types = [ct for ct in sorted(set(all_cell_types)) if ct in celltypes_to_draw]
            if not all_cell_types:
                print(f"No requested cell types present for simulation {simulation_index}; skipping.")
                continue
        else:
            all_cell_types = sorted(list(all_cell_types))

        cells_per_page = 4
        num_pages = math.ceil(len(all_cell_types) / cells_per_page)

        for page in range(num_pages):
            start_idx = page * cells_per_page
            end_idx = min(start_idx + cells_per_page, len(all_cell_types))
            page_cell_types = all_cell_types[start_idx:end_idx]

            rows = math.ceil(len(page_cell_types) / 2)
            cols = min(len(page_cell_types), 2)
            fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 8*rows))
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1 or cols == 1:
                axes = axes.reshape(-1, 1) if cols == 1 else axes.reshape(1, -1)

            for i, cell_type in enumerate(page_cell_types):
                row_idx, col_idx = i // 2, i % 2
                ax = axes[row_idx, col_idx]
                fpr_common = np.linspace(0, 1, 101)
                labels_leg = {}

                for method in methods:
                    tpr_interp_list, aucs = [], []
                    for simulation_seed in simulation_seeds:
                        if method['name'] == 'CYGNET':
                            file_path = os.path.join(method['dir'], f"sim_{simulation_index}_{simulation_seed}_lmm_fdr_all_results.csv")
                        else:
                            file_path = os.path.join(method['dir'], f"sim_{simulation_index}_{simulation_seed}_cside_fdr_all_results.csv")
                        if not os.path.exists(file_path):
                            continue
                        try:
                            results = pd.read_csv(file_path)
                            gene_col = 'Gene' if 'Gene' in results.columns else 'gene_id'
                            celltype_col = 'Celltype' if 'Celltype' in results.columns else 'celltype'
                            ct_data = results[results[celltype_col] == cell_type]
                            if ct_data.empty:
                                continue
                            gene_to_score = {row[gene_col]: 1 - row['FDR_p_value']
                                             for _, row in ct_data.iterrows()}
                            all_tested_pairs_this_celltype = {
                                pair for pair in all_gene_celltype_pairs if pair[1] == cell_type
                            }
                            labels = []
                            scores = []
                            true_positives_this_celltype = {
                                pair for pair in true_positive_sets[simulation_seed] if pair[1] == cell_type
                            }
                            if all_true_positives:
                                for gene, _ in all_tested_pairs_this_celltype:
                                    is_tp = (gene, cell_type) in true_positives_this_celltype
                                    score = gene_to_score.get(gene, 0)
                                    labels.append(1 if is_tp else 0)
                                    scores.append(score)
                            else:
                                for gene in set(ct_data[gene_col]):
                                    is_tp = (gene, cell_type) in true_positives_this_celltype
                                    score = gene_to_score[gene]
                                    labels.append(1 if is_tp else 0)
                                    scores.append(score)
                            if sum(labels) > 0:
                                fpr, tpr, _ = roc_curve(labels, scores)
                                roc_auc = auc(fpr, tpr)
                                if aggregate == 'all':
                                    label = f"{method['name']} (AUC={roc_auc:.3f})" if (simulation_seed == simulation_seeds[0]) else None
                                    l = ax.plot(fpr, tpr, color=method['color'], alpha=0.37, lw=1.3, label=label)
                                    labels_leg[method['name']] = l[0]
                                elif aggregate == 'mean':
                                    interp_func = interp1d(fpr, tpr, bounds_error=False, fill_value=(0,1))
                                    tpr_interp = interp_func(fpr_common)
                                    tpr_interp_list.append(tpr_interp)
                                    aucs.append(roc_auc)
                        except Exception as e:
                            print(f"Error for index {simulation_index} ct={cell_type}, method={method['name']}, seed={simulation_seed}: {e}")
                    if aggregate == 'mean' and tpr_interp_list:
                        tpr_interp_arr = np.stack(tpr_interp_list, axis=0)
                        mean_tpr = np.mean(tpr_interp_arr, axis=0)
                        lower = np.percentile(tpr_interp_arr, 100*alpha, axis=0)
                        upper = np.percentile(tpr_interp_arr, 100*(1-alpha), axis=0)
                        mean_auc = np.mean(aucs)
                        l = ax.plot(fpr_common, mean_tpr, color=method['color'], lw=2.5,
                                 label=f"{method['name']} Mean (AUC={mean_auc:.3f})")
                        ax.fill_between(fpr_common, lower, upper, color=method['color'], alpha=0.24)
                        labels_leg[method['name']] = l[0]

                ax.plot([0, 1], [0, 1], 'k--', lw=1)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_title(f'Cell Type: {cell_type}')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                handles, labels_ = ax.get_legend_handles_labels()
                unique = []
                seen = set()
                for h, l in zip(handles, labels_):
                    if l and l not in seen:
                        unique.append((h, l))
                        seen.add(l)
                if unique:
                    ax.legend([h for h, _ in unique], [l for _, l in unique], loc='lower right', fontsize=8)
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for i in range(len(page_cell_types), rows * cols):
                row_idx = i // 2
                col_idx = i % 2
                axes[row_idx, col_idx].axis('off')

            eval_mode = "All True Positives" if all_true_positives else "Only Tested Genes"
            aggmode = 'All seeds' if aggregate=='all' else f'Mean+{int(ci*100)}%CI'
            plt.suptitle(
                f'ROC Curves ({aggmode}) - Sim {simulation_index} ({eval_mode})',
                fontsize=14
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            file_suffix = "all_tp" if all_true_positives else "tested_only"
            output_file = os.path.join(
                output_dir,
                f'roc_grid_{aggmode.lower().replace("+", "")}_sim{simulation_index}_page{page+1}_{file_suffix}.png'
            )
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"ROC grid ({aggmode}) for simulation {simulation_index}, page {page+1} saved.")
