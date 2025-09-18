import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

# LaTeX-friendly labels for parameters
latex_param_labels = {
    "sigma": r"$\sigma$",
    "X": r"$X$",
    "vMax": r"$v_{\max}$",
    "d": r"$d$"
}


def plot_sensitivity_results(param, date_str=None):
    """Plot sensitivity analysis results for optimization with F, c, d variables"""

    # Use today's date if not specified
    if date_str is None:
        date_str = datetime.today().strftime('%Y-%m-%d')

    folder = os.path.join("results", "optimization", date_str)
    print(f"📁 Reading results from folder: {folder}")

    # Load CSV
    csv_file = os.path.join(folder, f"sensitivity_{param}.csv")
    df = pd.read_csv(csv_file)
    print(f"📊 Loaded {len(df)} data points for {param}")

    # Set plotting style
    sns.set_context("talk", font_scale=1.2)
    sns.set_style("whitegrid")

    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'lines.markersize': 8,
        'lines.linewidth': 2.5,
        'legend.fontsize': 14,
        'legend.title_fontsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'figure.titlesize': 22,
        'figure.dpi': 300,
    })

    # Define color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Helper function to set dynamic y-limits
    def set_dynamic_ylim(ax, values, min_range=0.1, epsilon=0.07):
        flat_values = np.array(pd.to_numeric(values, errors='coerce')).flatten()
        flat_values = flat_values[~np.isnan(flat_values)]
        if len(flat_values) == 0:
            return
        ymin, ymax = np.min(flat_values), np.max(flat_values)
        if ymax - ymin < min_range:
            center = (ymax + ymin) / 2
            ymin = center - (min_range / 2) - epsilon
            ymax = center + (min_range / 2) + epsilon
        else:
            ymin -= epsilon
            ymax += epsilon
        extra_padding = 0.02 * (ymax - ymin)
        ax.set_ylim(ymin - extra_padding, ymax + extra_padding)

    # Beautify plot function
    def beautify_line_plot(ax, x, y, ylabel, color, add_markers=True):
        """Create a beautiful line plot with consistent styling"""
        # Filter out NaN values
        mask = ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]

        if add_markers:
            ax.plot(x_clean, y_clean, '-o', color=color, markersize=8, linewidth=2.5, alpha=0.8)
        else:
            ax.plot(x_clean, y_clean, '-', color=color, linewidth=2.5, alpha=0.8)

        param_label = latex_param_labels.get(param, param)
        ax.set_xlabel(param_label, fontsize=20, weight='bold')
        ax.set_ylabel(ylabel, fontsize=18)
        ax.grid(True, linestyle='--', alpha=0.3)
        set_dynamic_ylim(ax, y_clean)
        ax.tick_params(axis='both', labelsize=16)

        # Add shaded area for successful runs
        if 'success' in df.columns:
            success_mask = df['success'] == True
            if success_mask.any():
                ax.fill_between(x[success_mask], ax.get_ylim()[0], ax.get_ylim()[1],
                                alpha=0.1, color='green', label='Successful')

    # --- Plot 1: Decision Variables (F, c, d) and Profit ---
    fig1, axs1 = plt.subplots(4, 1, figsize=(14, 16))
    plt.subplots_adjust(hspace=0.4)

    beautify_line_plot(axs1[0], df[param], df['optimal_f'], r'$F$ (Fixed Fee)', colors[0])
    axs1[0].set_title('Decision Variables and Profit', fontsize=20, weight='bold', pad=20)

    beautify_line_plot(axs1[1], df[param], df['optimal_c'], r'$c$ (Per-unit Price)', colors[1])
    beautify_line_plot(axs1[2], df[param], df['optimal_d'], r'$d$ (Discount Rate)', colors[2])
    beautify_line_plot(axs1[3], df[param], df['optimal_pi'], r'$\pi$ (Profit)', colors[3])

    # Add success indicators
    if 'success' in df.columns:
        for ax in axs1:
            failed_mask = df['success'] == False
            if failed_mask.any():
                y_pos = ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.scatter(df[param][failed_mask], [y_pos] * sum(failed_mask),
                           marker='x', color='red', s=50, zorder=10, label='Failed')

    fig1.suptitle(f'Optimization Results vs {latex_param_labels.get(param, param)}',
                  fontsize=22, weight='bold')
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(os.path.join(folder, f"sensitivity_{param}_decisions_profit.pdf"), bbox_inches='tight')
    plt.show()

    # --- Plot 2: Beta Fractions ---
    fig2, axs2 = plt.subplots(3, 1, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.4)

    beautify_line_plot(axs2[0], df[param], df['beta_P'], r'$\beta^P$ (Platinum)', colors[0])
    axs2[0].set_title('Customer Segment Fractions', fontsize=20, weight='bold', pad=20)

    beautify_line_plot(axs2[1], df[param], df['beta_R'], r'$\beta^R$ (Regular)', colors[1])
    beautify_line_plot(axs2[2], df[param], df['beta_N'], r'$\beta^N$ (Non-users)', colors[2])

    # Add sum check line
    beta_sum = df['beta_P'] + df['beta_R'] + df['beta_N']
    for ax in axs2:
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Sum=1')

    fig2.suptitle(f'Beta Fractions vs {latex_param_labels.get(param, param)}',
                  fontsize=22, weight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(os.path.join(folder, f"sensitivity_{param}_beta_fractions.pdf"), bbox_inches='tight')
    plt.show()

    # --- Plot 3: Delta Values ---
    fig3, axs3 = plt.subplots(2, 1, figsize=(14, 8))
    plt.subplots_adjust(hspace=0.4)

    beautify_line_plot(axs3[0], df[param], df['delta_P'], r'$\delta^P$ (Platinum Demand)', colors[4])
    axs3[0].set_title('Demand Quantities', fontsize=20, weight='bold', pad=20)

    beautify_line_plot(axs3[1], df[param], df['delta_R'], r'$\delta^R$ (Regular Demand)', colors[5])

    fig3.suptitle(f'Delta Values vs {latex_param_labels.get(param, param)}',
                  fontsize=22, weight='bold')
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig(os.path.join(folder, f"sensitivity_{param}_delta_values.pdf"), bbox_inches='tight')
    plt.show()

    # --- Plot 4: Combined Overview ---
    fig4, axs4 = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: Decision variables
    beautify_line_plot(axs4[0, 0], df[param], df['optimal_f'], r'$F$', colors[0], add_markers=False)
    beautify_line_plot(axs4[0, 1], df[param], df['optimal_c'], r'$c$', colors[1], add_markers=False)
    beautify_line_plot(axs4[0, 2], df[param], df['optimal_d'], r'$d$', colors[2], add_markers=False)

    # Bottom row: Profit and key fractions
    beautify_line_plot(axs4[1, 0], df[param], df['optimal_pi'], r'$\pi$', colors[3], add_markers=False)

    # Stack plot for betas
    axs4[1, 1].stackplot(df[param], df['beta_N'], df['beta_R'], df['beta_P'],
                         labels=[r'$\beta^N$', r'$\beta^R$', r'$\beta^P$'],
                         colors=['#c7e9b4', '#7fcdbb', '#41b6c4'], alpha=0.8)
    axs4[1, 1].set_xlabel(latex_param_labels.get(param, param), fontsize=18, weight='bold')
    axs4[1, 1].set_ylabel('Customer Fractions', fontsize=16)
    axs4[1, 1].legend(loc='best', fontsize=12)
    axs4[1, 1].grid(True, linestyle='--', alpha=0.3)

    # Deltas comparison
    axs4[1, 2].plot(df[param], df['delta_P'], '-', color=colors[4], linewidth=2.5, label=r'$\delta^P$')
    axs4[1, 2].plot(df[param], df['delta_R'], '-', color=colors[5], linewidth=2.5, label=r'$\delta^R$')
    axs4[1, 2].set_xlabel(latex_param_labels.get(param, param), fontsize=18, weight='bold')
    axs4[1, 2].set_ylabel('Demand', fontsize=16)
    axs4[1, 2].legend(loc='best', fontsize=12)
    axs4[1, 2].grid(True, linestyle='--', alpha=0.3)

    fig4.suptitle(f'Comprehensive Sensitivity Analysis: {latex_param_labels.get(param, param)}',
                  fontsize=22, weight='bold')
    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    fig4.savefig(os.path.join(folder, f"sensitivity_{param}_overview.pdf"), bbox_inches='tight')
    plt.show()

    # --- Print Summary Statistics ---
    print("\n" + "=" * 60)
    print(f"SUMMARY STATISTICS FOR {param}")
    print("=" * 60)

    print("\nOptimal Values Range:")
    print(f"  F: [{df['optimal_f'].min():.4f}, {df['optimal_f'].max():.4f}]")
    print(f"  c: [{df['optimal_c'].min():.4f}, {df['optimal_c'].max():.4f}]")
    print(f"  d: [{df['optimal_d'].min():.4f}, {df['optimal_d'].max():.4f}]")
    print(f"  π: [{df['optimal_pi'].min():.4f}, {df['optimal_pi'].max():.4f}]")

    if 'success' in df.columns:
        success_rate = df['success'].sum() / len(df) * 100
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        print(f"Failed runs: {(~df['success']).sum()} out of {len(df)}")

    print("\nBeta Fractions Check (should sum to 1):")
    beta_sums = df['beta_P'] + df['beta_R'] + df['beta_N']
    print(f"  Min sum: {beta_sums.min():.6f}")
    print(f"  Max sum: {beta_sums.max():.6f}")
    print(f"  Mean sum: {beta_sums.mean():.6f}")

    print(f"\n✅ All plots saved to: {folder}")


# Usage examples:
if __name__ == "__main__":
    # Plot for different parameters
    plot_sensitivity_results(param="sigma", date_str="2025-09-18")
    # plot_sensitivity_results(param="X", date_str="2025-09-17")
    # plot_sensitivity_results(param="vMax", date_str="2025-09-17")

    # Or use today's date automatically
    # plot_sensitivity_results(param="sigma")