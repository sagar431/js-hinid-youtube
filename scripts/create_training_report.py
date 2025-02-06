import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_training_plots():
    # Read metrics from CSV
    metrics_file = Path("logs/train/runs").glob("*/csv/version_0/metrics.csv")
    metrics_file = sorted(metrics_file)[-1]  # Get latest run
    df = pd.read_csv(metrics_file)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = [12, 6]
    
    # Create accuracy plot
    plt.figure()
    sns.lineplot(data=df, x='step', y='train/acc', label='Train', marker='o')
    sns.lineplot(data=df, x='step', y='val/acc', label='Validation', marker='o')
    plt.title('Model Accuracy Over Time', pad=20)
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create loss plot
    plt.figure()
    sns.lineplot(data=df, x='step', y='train/loss', label='Train', marker='o')
    sns.lineplot(data=df, x='step', y='val/loss', label='Validation', marker='o')
    plt.title('Model Loss Over Time', pad=20)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create test metrics table
    test_metrics = df[['test/loss', 'test/acc']].dropna().iloc[-1]
    
    # Create performance summary table
    final_metrics = pd.DataFrame({
        'Metric': ['Training Loss', 'Validation Loss', 'Test Loss', 
                  'Training Accuracy', 'Validation Accuracy', 'Test Accuracy'],
        'Value': [
            df['train/loss'].iloc[-1],
            df['val/loss'].iloc[-1],
            test_metrics['test/loss'],
            df['train/acc'].iloc[-1],
            df['val/acc'].iloc[-1],
            test_metrics['test/acc']
        ]
    })
    
    metrics_table = "## Performance Summary\n\n"
    metrics_table += "| Metric | Value |\n|--------|-------|\n"
    for _, row in final_metrics.iterrows():
        metrics_table += f"| {row['Metric']} | {row['Value']:.4f} |\n"

    # Create the report
    report = f"""# Model Training Report 📊

## Training Progress
### Accuracy
![](./accuracy_plot.png)

### Loss
![](./loss_plot.png)

{metrics_table}

### Training Details
- Total Steps: {len(df)}
- Best Validation Accuracy: {df['val/acc'].max():.4f}
- Best Test Accuracy: {test_metrics['test/acc']:.4f}
- Final Training Loss: {df['train/loss'].iloc[-1]:.4f}

### Model Performance Analysis
- The model {'improved' if df['val/acc'].iloc[-1] > df['val/acc'].iloc[0] else 'declined'} during training
- Validation accuracy {'tracked closely with' if abs(df['train/acc'].iloc[-1] - df['val/acc'].iloc[-1]) < 0.1 else 'diverged from'} training accuracy
- {'No significant overfitting detected' if abs(df['train/acc'].iloc[-1] - df['val/acc'].iloc[-1]) < 0.1 else 'Potential overfitting detected'}
"""

    # Save the report
    with open('report.md', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    create_training_plots() 