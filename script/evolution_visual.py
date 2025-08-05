import json
import matplotlib.pyplot as plt
import numpy as np
import os

JSON_PATH = "C:\\Users\\Andrew\\Downloads\\jsonformatter.json"


def load_evolution_data(json_file_path):
    """Load evolution data from JSON file."""
    with open(json_file_path, "r") as f:
        return json.load(f)


def is_cooperative_action(action, points):
    if action == "COOPERATE":
        return True
    elif action == "DEFECT":
        return False


def convert_points_scale(points):
    # To ensure reputation and points are on the same scale
    point_mapping = {0: 0, 1: 0.3333, 3: 1, 5: 1.6667}
    return point_mapping.get(points, points)


def extract_model_data(step_data, step_number):
    if not step_data.get("match_records"):
        print(f"Step {step_number}: No match records")
        return None

    # Collect all models from all matches in this step
    all_models = {}
    total_matches = 0

    for match in step_data["match_records"]:
        if not match:
            continue

        total_matches += 1
        for player in match:
            name = player.get("name", "Unknown")
            points = convert_points_scale(player.get("points", 0))
            action = player.get("action", "UNKNOWN")

            if name not in all_models:
                all_models[name] = {"points": [], "actions": [], "raw_points": []}

            all_models[name]["points"].append(points)
            all_models[name]["actions"].append(action)
            all_models[name]["raw_points"].append(player.get("points", 0))

    print(
        f"Step {step_number}: Found {len(all_models)} unique models in {total_matches} matches"
)

    # Calculate metrics for each model
    model_data = []
    for name, data in all_models.items():
        if data["points"]:  # Only include models with data
            avg_points = np.mean(data["points"])

            # Calculate cooperation rate based on actions
            cooperation_count = sum(
                1 for i, action in enumerate(data["actions"]) if action == "COOPERATE"
            )
            cooperation_rate = (
                cooperation_count / len(data["actions"]) if data["actions"] else 0
            )

            # Calculate betrayal rate (inverse of cooperation rate)
            betrayal_rate = 1 - cooperation_rate

            model_data.append(
                {
                    "name": name,
                    "avg_points": avg_points,
                    "cooperation_rate": cooperation_rate,
                    "betrayal_rate": betrayal_rate,
                    "action_count": len(data["actions"]),
                }
            )

            print(
                f"  {name}: {len(data['actions'])} actions, {cooperation_count} cooperative, cooperation rate: {cooperation_rate:.2f}, betrayal rate: {betrayal_rate:.2f}, avg points: {avg_points:.3f}"
            )

    return model_data if model_data else None


def create_cooperation_visualization(model_data, step_number, output_dir):
    """Create cooperation rate vs points visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))

    model_points = [m["avg_points"]for m in model_data]
    model_reputations = [m["cooperation_rate"] for m in model_data]
    model_names = [m["name"] for m in model_data]

    # Calculate Pearson correlation with theoretical line: y = 0.3333 + (2/3)*x
    if len(model_data) > 1:  # Need at least 2 points for correlation
        theoretical_points = [
            0.3333 + (2 / 3) * reputation for reputation in model_reputations
        ]
        correlation = np.corrcoef(theoretical_points, model_points)[0, 1]
        correlation = correlation if not np.isnan(correlation) else 0
    else:
        correlation = 0

    # Create theoretical line
    # When all models defect, cooperation rate is 0 and both get 0.3333 point
    # When all models cooperate, cooperation rate is 1 and both get 1 point
    line_range = np.linspace(0, 1, 100)
    theoretical_points = 0.3333 + (2 / 3) * line_range
    ax.plot(
        line_range,
        theoretical_points,
        "k--",
        alpha=0.5,
        label="Theoretical cooperation",
        linewidth=2,
)

    # Add trend line for actual data
    if len(model_data) > 1:
        z = np.polyfit(model_reputations, model_points, 1)
        p = np.poly1d(z)
        trend_line = p(line_range)
        ax.plot(
            line_range,
            trend_line,
            "red",
            linestyle="-",
            linewidth=2,
            alpha=0.7,
            label=f"Actual trend (slope: {z[0]:.3f})",
    )

    # Plot each model
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    for i, (points, reputation, name) in enumerate(
        zip(model_points, model_reputations, model_names)
    ):
        display_name = (
            name.split("(")[0][:15] + "..." if len(name) > 18 else name.split("(")[0]
        )
        ax.scatter(
            reputation,
            points,
            c=[colors[i]],
            s=100,
            alpha=0.7,
            label=display_name,
            edgecolors="black",
            linewidth=0.5,
    )

    # Design plot
    ax.set_xlabel("Reputation (Cooperation Rate)", fontsize=12)
    ax.set_ylabel("Points (Scaled: 0.3333, 1, 1.6667)", fontsize=12)
    ax.set_title(
        f"Step {step_number}: Cooperation Rate vs Points\nCorrelation with theoretical cooperation: {correlation:.4f}",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.75)

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, f"step_{step_number:03d}_cooperation.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cooperation visualization for step {step_number} to {output_file}")
    print(f"Correlation with theoretical cooperation: {correlation:.4f}")


def create_step_visualizations(step_data, step_number, output_dir):
    """Create both cooperation and betrayal visualizations for a single step."""
    model_data = extract_model_data(step_data, step_number)

    if not model_data:  # Skip if no valid data
        print(f"Step {step_number}: No valid model data found")
        return

    # Create both visualizations
    create_cooperation_visualization(model_data, step_number, output_dir)
    print()


def create_all_visualizations(json_file_path, output_dir="evolution_plots"):
    """Create visualizations for all steps in the evolution data."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_evolution_data(json_file_path)

    print(f"Creating visualizations for {len(data)} steps...")

    # Create visualization for each step
    for step_data in data:
        step_number = step_data.get("step", 0)
        create_step_visualizations(step_data, step_number, output_dir)

    print(f"All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file not found at {JSON_PATH}")
        exit()

    create_all_visualizations(JSON_PATH)

    print("Evolution visualization complete!")
