import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def analyze_evolution_trends(json_file_path):
    """Analyze trends across all evolution steps."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    steps = []
    avg_cooperation_rates = []
    avg_points = []
    model_counts = []
    cooperation_correlations = []  # Within-step correlations (model-level)

    # Store data for overall correlation calculation
    all_step_coop_rates = []
    all_step_avg_points = []

    for step_data in data:
        step_num = step_data.get('step', 0)
        steps.append(step_num)

        # Analyze this step's data
        cooperation_rates = []
        points_list = []
        models_in_step = set()
        step_models = {}

        for match in step_data.get('match_records', []):
            if not match:
                continue

            for player in match:
                name = player.get('name', 'Unknown')
                models_in_step.add(name)

                # Determine cooperation
                is_cooperative = (player.get('action') == 'COOPERATE')
                cooperation_rate = 1.0 if is_cooperative else 0.0
                cooperation_rates.append(cooperation_rate)

                # Map raw points to scaled values
                raw_points = player.get('points', 0)
                point_mapping = {0: 0, 1: 0.3333, 3: 1, 5: 1.6667}
                converted_points = point_mapping.get(raw_points, raw_points)
                points_list.append(converted_points)

                # Track per-model stats
                if name not in step_models:
                    step_models[name] = {'cooperation': [], 'points': []}
                step_models[name]['cooperation'].append(cooperation_rate)
                step_models[name]['points'].append(converted_points)

        # Compute per-step averages
        step_avg_coop = np.mean(cooperation_rates) if cooperation_rates else 0
        step_avg_points = np.mean(points_list) if points_list else 0
        avg_cooperation_rates.append(step_avg_coop)
        avg_points.append(step_avg_points)
        model_counts.append(len(models_in_step))
        all_step_coop_rates.append(step_avg_coop)
        all_step_avg_points.append(step_avg_points)

        # Compute within-step correlation
        if len(step_models) > 1:
            model_coop_rates = [np.mean(md['cooperation']) for md in step_models.values()]
            model_avg_points = [np.mean(md['points']) for md in step_models.values()]
            corr = np.corrcoef(model_coop_rates, model_avg_points)[0, 1]
            cooperation_correlations.append(corr if not np.isnan(corr) else 0)
        else:
            cooperation_correlations.append(0)

    # Overall correlation
    overall_corr = 0
    if len(all_step_coop_rates) > 1:
        overall_corr = np.corrcoef(all_step_coop_rates, all_step_avg_points)[0, 1]
        overall_corr = overall_corr if not np.isnan(overall_corr) else 0

    return steps, avg_cooperation_rates, avg_points, model_counts, cooperation_correlations, overall_corr


def create_trend_analysis(json_file_path, output_dir='evolution_plots'):
    """Create and save trend analysis plots based on given JSON data."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    steps, coop_rates, avg_pts, model_counts, coop_corrs, overall_corr = \
        analyze_evolution_trends(json_file_path)

    # Reload for detailed per-point stats
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Average Cooperation Rate
    ax1.plot(steps, coop_rates, marker='o')
    ax1.set(title='Cooperation Rate Over Time', xlabel='Evolution Step', ylabel='Avg Cooperation Rate')
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Panel 2: Reputation-Points Correlation
    ax2.plot(steps, coop_corrs, marker='d')
    ax2.set(title='Reputation-Points Correlation', xlabel='Evolution Step', ylabel='Pearson Correlation')
    ax2.axhline(0, linestyle='--')
    ax2.set_ylim(-1, 1)
    ax2.grid(alpha=0.3)

    # Panel 3: Reputation vs Performance Scatter
    scatter = ax3.scatter(coop_rates, avg_pts, c=steps, cmap='viridis')
    ax3.set(title='Reputation vs Performance', xlabel='Avg Reputation', ylabel='Avg Points')
    ax3.grid(alpha=0.3)
    # Theoretical line
    x_line = np.linspace(0, 1, 100)
    y_line = 0.3333 + (2/3)*x_line
    ax3.plot(x_line, y_line, linestyle='--')
    fig.colorbar(scatter, ax=ax3, label='Step')

    # Panel 4: Learning Curve
    step_means, step_stds = [], []
    for step_data in data:
        pts = [ {0:0,1:0.3333,3:1,5:1.6667}.get(p['points'], p['points'])
                for match in step_data.get('match_records', []) if match
                for p in match ]
        step_means.append(np.mean(pts) if pts else 0)
        step_stds.append(np.std(pts) if pts else 0)
    ax4.plot(steps, step_means, marker='o')
    if len(steps) > 2:
        z = np.polyfit(steps, step_means, 1)
        ax4.plot(steps, np.poly1d(z)(steps), linestyle='--')
    ax4.set(title='System Learning Curve', xlabel='Evolution Step', ylabel='Avg Points')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'evolution_analysis.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Analyze evolutionary cooperation trends and generate plots.')
    parser.add_argument('json_file', help='Path to JSON data file')
    parser.add_argument('--output-dir', default='evolution_plots',
                        help='Directory to save output plots')
    args = parser.parse_args()

    create_trend_analysis(args.json_file, args.output_dir)


if __name__ == '__main__':
    main()