import json
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_funsd_sample(json_path, image_path, output_path=None):
    """
    Visualize a single FUNSD dataset sample with its annotations.
    
    Args:
        json_path: Path to the JSON annotation file
        image_path: Path to the corresponding image file
        output_path: Optional path to save the visualization
    """
    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load the image
    image = Image.open(image_path)
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot annotated image
    ax2.imshow(image)
    ax2.set_title('Annotated Image')
    ax2.axis('off')
    
    # Colors for different labels
    colors = {
        'question': 'red',
        'answer': 'green',
        'header': 'blue',
        'other': 'gray'
    }
    
    # Draw annotations
    for item in data['form']:
        label = item['label']
        color = colors.get(label, 'gray')
        
        # Draw box for the entire item
        box = item['box']
        rect = patches.Rectangle(
            (box[0], box[1]), 
            box[2] - box[0], 
            box[3] - box[1], 
            linewidth=2, 
            edgecolor=color, 
            facecolor='none',
            alpha=0.7
        )
        ax2.add_patch(rect)
        
        # Add label text
        ax2.text(
            box[0], box[1] - 5,
            f"{label} (id: {item['id']})",
            color=color,
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        
        # Draw boxes for individual words
        for word_info in item['words']:
            if isinstance(word_info, dict):
                word_box = word_info['box']
                word_rect = patches.Rectangle(
                    (word_box[0], word_box[1]),
                    word_box[2] - word_box[0],
                    word_box[3] - word_box[1],
                    linewidth=1,
                    edgecolor=color,
                    facecolor='none',
                    alpha=0.3
                )
                ax2.add_patch(word_rect)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=color, label=label)
        for label, color in colors.items()
    ]
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add dataset statistics
    stats_text = f"Total items: {len(data['form'])}\n"
    label_counts = {}
    for item in data['form']:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in label_counts.items():
        stats_text += f"{label}: {count}\n"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and display
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_path}")
    
    plt.show()

def analyze_linking(json_data):
    """
    Analyze and display the linking information between elements.
    
    Args:
        json_data: Loaded JSON data
    """
    print("\nAnalyzing linking relationships:")
    print("-" * 50)
    
    for item in json_data['form']:
        if item['linking']:
            label = item['label']
            item_id = item['id']
            print(f"\nItem {item_id} ({label}):")
            print(f"Text: {item['text']}")
            print("Linked to:")
            for link in item['linking']:
                for linked_id in link:
                    if linked_id != item_id:
                        linked_item = next(x for x in json_data['form'] if x['id'] == linked_id)
                        print(f"  - Item {linked_id} ({linked_item['label']}): {linked_item['text']}")

if __name__ == "__main__":
    # Example usage
    json_path = "dataset/testing_data/annotations/82491256.json"  # Update this path
    image_path = "dataset/testing_data/images/82491256.png"      # Update this path
    output_path = "visualization.png"           # Optional: path to save visualization
    
    # Visualize the sample
    visualize_funsd_sample(json_path, image_path, output_path)
    
    # Analyze linking relationships
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    analyze_linking(json_data)