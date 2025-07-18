from typing import List, Tuple, Dict
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

def cluster_image_sizes(image_sizes: List[Tuple[int, int]],
                        n_bins: int = 20, 
                        return_stats: bool = False):
    """
    Cluster image sizes into N bins using k-means clustering.
    
    Args:
        image_sizes: List of tuples containing image sizes as (width, height)
        n_bins: Number of bins to cluster into (default: 20)
        return_stats: Whether to return statistics dictionary (default: False)
        
    Returns:
        unique_sizes containing:
        - cluster_sizes: List of representative cluster sizes as (width, height) tuples
        - cluster_mapping: Dict where keys are cluster sizes and values are lists of indices
        - stats: Dict with statistics for each bin (only if return_stats=True)
    """
    if not image_sizes:
        if return_stats:
            return [], {}, {}
        return [], {}
    
    # Use minimum of requested bins and available unique sizes
    unique_sizes = list(set(image_sizes))
    n_clusters = min(n_bins, len(unique_sizes))
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(image_sizes)
    
    # Get cluster centers and find representative sizes
    cluster_sizes = []
    cluster_mapping = {}
    
    for i in range(n_clusters):
        # Find all indices that belong to this cluster
        cluster_indices = np.where(cluster_labels == i)[0].tolist()
        
        if not cluster_indices:
            continue
        
        # Get cluster center
        cluster_center = kmeans.cluster_centers_[i]
        center_width, center_height = int(round(cluster_center[0])), int(round(cluster_center[1]))
        
        # Find the actual size closest to the cluster center among items in this cluster
        min_distance = float('inf')
        representative_size = image_sizes[cluster_indices[0]]
        
        for idx in cluster_indices:
            size = image_sizes[idx]
            distance = np.sqrt((size[0] - center_width)**2 + (size[1] - center_height)**2)
            if distance < min_distance:
                min_distance = distance
                representative_size = size
        
        cluster_sizes.append(representative_size)
        cluster_mapping[representative_size] = cluster_indices
    
    if not return_stats:
        return cluster_sizes, cluster_mapping
    
    # Calculate statistics for each cluster
    stats = {}
    total_images = len(image_sizes)
    
    for cluster_size, indices in cluster_mapping.items():
        cluster_image_sizes = [image_sizes[i] for i in indices]
        
        # Calculate statistics
        total_pixels_list = [w * h for w, h in cluster_image_sizes]
        avg_pixels = sum(total_pixels_list) / len(total_pixels_list)
        
        # Get size distribution within cluster
        size_counts = {}
        for size in cluster_image_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        
        stats[cluster_size] = {
            'count': len(indices),
            'avg_pixels': avg_pixels,
            'dimensions': cluster_size,
            'size_distribution': size_counts,
            'percentage': (len(indices) / total_images) * 100
        }
    
    cluster_stats = {
        'total_images': total_images,
        'total_bins': len(cluster_sizes),
        'original_unique_sizes': len(unique_sizes),
        'clustering_method': f'k-means with {n_clusters} clusters',
        'bins': stats
    }
    
    return cluster_sizes, cluster_mapping, cluster_stats

def plot_cluster_statistics(cluster_stats: Dict, 
                            output_folder: 
                            str, filename: str = "cluster_statistics.png"):
    """
    Generate a bar plot showing the cluster statistics.
    
    Args:
        cluster_stats: Statistics dictionary from cluster_image_sizes with return_stats=True
        output_folder: Path to folder where the plot will be saved
        filename: Name of the output file (default: "cluster_statistics.png")
    """
    if not cluster_stats or 'bins' not in cluster_stats:
        print("No cluster statistics data available to plot")
        return
    
    # Extract data for plotting
    cluster_labels = []
    counts = []
    percentages = []
    avg_pixels = []
    
    # Sort clusters by count (descending) for better visualization
    sorted_clusters = sorted(cluster_stats['bins'].items(), key=lambda x: x[1]['count'], reverse=True)
    
    for cluster_size, data in sorted_clusters:
        width, height = cluster_size
        cluster_labels.append(f"{width}x{height}")
        counts.append(data['count'])
        percentages.append(data['percentage'])
        avg_pixels.append(data['avg_pixels'])
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Count distribution
    bars1 = ax1.bar(range(len(cluster_labels)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_xlabel('Cluster Size', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Image Count per Cluster', fontsize=14)
    ax1.set_xticks(range(len(cluster_labels)))
    ax1.set_xticklabels(cluster_labels, rotation=45, ha='right')
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # Add value labels on bars
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Percentage distribution
    bars2 = ax2.bar(range(len(cluster_labels)), percentages, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax2.set_xlabel('Cluster Size', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Percentage Distribution per Cluster', fontsize=14)
    ax2.set_xticks(range(len(cluster_labels)))
    ax2.set_xticklabels(cluster_labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, percentage in zip(bars2, percentages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(percentages)*0.01,
                f"{percentage:.1f}%", ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Average pixels per cluster
    bars3 = ax3.bar(range(len(cluster_labels)), avg_pixels, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax3.set_xlabel('Cluster Size', fontsize=12)
    ax3.set_ylabel('Average Pixels', fontsize=12)
    ax3.set_title('Average Pixels per Cluster', fontsize=14)
    ax3.set_xticks(range(len(cluster_labels)))
    ax3.set_xticklabels(cluster_labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, avg_pix in zip(bars3, avg_pixels):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_pixels)*0.01,
                f"{int(avg_pix)}", ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Summary statistics as text
    ax4.axis('off')
    summary_text = f"""
    Clustering Summary:
    
    Total Images: {cluster_stats['total_images']:,}
    Original Unique Sizes: {cluster_stats['original_unique_sizes']:,}
    Clustered into: {cluster_stats['total_bins']} bins
    Method: {cluster_stats['clustering_method']}
    
    Largest Cluster: {cluster_labels[0]} ({counts[0]:,} images, {percentages[0]:.1f}%)
    Smallest Cluster: {cluster_labels[-1]} ({counts[-1]:,} images, {percentages[-1]:.1f}%)
    
    Average Images per Cluster: {cluster_stats['total_images'] / cluster_stats['total_bins']:.1f}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Overall title
    fig.suptitle(f'Cluster Analysis Dashboard\n{cluster_stats["clustering_method"]}', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the plot
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    full_path = output_path / filename
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cluster statistics plot saved to: {full_path}")