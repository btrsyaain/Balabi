import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def process_file(filename):
    """
    Process data file, extracting non-zero values from column 1
    """
    with open(filename, 'r') as f:
        data_column1 = []
        for line in f:
            row = np.fromstring(line.strip(), sep=' ')
            if row[1] != 0:
                data_column1.append(row[1])
    
    return np.array(data_column1)

def compare_data_distribution(file1, file2, output_dir='./'):
    """
    Compare data distribution for column 1 between two files
    """
    data1 = process_file(file1)
    data2 = process_file(file2)
    
    # Statistical summary
    print(f"{file1} Column 1 Statistics:")
    print(f"Count: {len(data1)}")
    print(f"Mean: {np.mean(data1):.4f}")
    print(f"Std Dev: {np.std(data1):.4f}")
    print(f"Min: {np.min(data1):.4f}")
    print(f"Max: {np.max(data1):.4f}")
    
    print(f"\n{file2} Column 1 Statistics:")
    print(f"Count: {len(data2)}")
    print(f"Mean: {np.mean(data2):.4f}")
    print(f"Std Dev: {np.std(data2):.4f}")
    print(f"Min: {np.min(data2):.4f}")
    print(f"Max: {np.max(data2):.4f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Histograms
    plt.subplot(2, 2, 1)
    plt.hist(data1, bins=30, alpha=0.7, color='blue', label=file1)
    plt.title(f'{file1} Column 1 Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.hist(data2, bins=30, alpha=0.7, color='red', label=file2)
    plt.title(f'{file2} Column 1 Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Boxplot
    plt.subplot(2, 2, 3)
    plt.boxplot([data1, data2], labels=[file1, file2])
    plt.title('Column 1 Boxplot Comparison')
    plt.ylabel('Value')
    
    # Q-Q Plot
    plt.subplot(2, 2, 4)
    stats.probplot(data1, dist="norm", plot=plt)
    plt.title(f'{file1} Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/column1_comparison.png', dpi=300)
    plt.close()

    # Statistical tests
    t_stat, p_value = stats.ttest_ind(data1, data2)
    print("\nTwo-sample t-test:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

# Usage
compare_data_distribution('./clean_sscore_record.txt', './poisoned_sscore_record.txt', './')