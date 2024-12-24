import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_analytical_visualization(json_path, image_path, output_path=None):
    """
    Create analytical visualization with bounding boxes and labels.
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
    ax2.set_title('Analytical Visualization')
    ax2.axis('off')
    
    # Colors for different labels
    colors = {
        'question': 'red',
        'answer': 'green',
        'header': 'blue',
        'other': 'gray'
    }
    
    # Draw annotations
    label_counts = {}
    for item in data['form']:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1
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
    
    # Add statistics
    stats_text = f"Document Statistics:\n\nTotal items: {len(data['form'])}\n"
    for label, count in label_counts.items():
        stats_text += f"{label}: {count}\n"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Analytical visualization saved to {output_path}")
    plt.close()

def create_document_visualization(json_path, image_path, output_path=None):
    """
    Create document-style visualization with color overlays.
    """
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load and convert image to RGBA
    img = Image.open(image_path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Create a blank overlay for all annotations
    final_img = img.copy()
    
    # Color mapping for different labels
    label_colors = {
        'header': 'salmon',
        'question': 'blue',
        'answer': 'green',
        'other': 'pink'
    }
    
    # Color map for overlay
    color_map = {
        'blue': (135, 206, 235),    # Light blue
        'green': (144, 238, 144),   # Light green
        'salmon': (250, 128, 114),  # Salmon
        'pink': (255, 182, 193)     # Light pink
    }
    
    # Draw annotations
    for item in data['form']:
        label = item['label']
        box = item['box']
        color_name = label_colors.get(label, 'gray')
        
        # Create overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        rgb_color = color_map.get(color_name, (128, 128, 128))
        rgba_color = (*rgb_color, 77)  # Alpha = 0.3 * 255
        
        draw.rectangle(box, fill=rgba_color)
        final_img = Image.alpha_composite(final_img, overlay)
    
    # Convert back to RGB for saving
    final_img = final_img.convert('RGB')
    
    if output_path:
        final_img.save(output_path)
        print(f"Document-style visualization saved to {output_path}")
    
    return final_img

def analyze_linking(json_data):
    """Print linking information between elements."""
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
    # Example usage - replace with your paths
    json_path = "dataset/testing_data/annotations/82491256.json"
    image_path = "dataset/testing_data/images/82491256.png" 
    
    # Create and save both visualizations
    create_analytical_visualization(json_path, image_path, "analytical_viz.png")
    create_document_visualization(json_path, image_path, "document_viz.png")
    
    # Analyze linking relationships
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    analyze_linking(json_data)
