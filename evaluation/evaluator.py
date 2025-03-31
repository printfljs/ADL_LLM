import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import os
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_predictions_split_activities(folder_name=None, dataset_name=None, true_labels_file=None, predicted_labels_file=None, output_file=None):
    """
    Multi-label evaluation method, accepts folder name, dataset name, and original file paths as parameters.
    Generates a comprehensive confusion matrix plot (0-1 range) containing all activities.
    """
    try:
        if true_labels_file is None or predicted_labels_file is None or output_file is None:
            if folder_name is None or dataset_name is None:
                raise ValueError("Folder name and dataset name or all file paths must be provided.")

            true_labels_file = os.path.join(folder_name, f'truth_labels_{dataset_name}.txt')
            predicted_labels_file = os.path.join(folder_name, f'predictions_{dataset_name}.txt')
            output_file = os.path.join(folder_name, f'evaluation_{dataset_name}.txt')
        print(f"True labels file: {true_labels_file}")
        print(f"Predicted labels file: {predicted_labels_file}")
        print(f"Output file: {output_file}")

        with open(true_labels_file, 'r') as f:
            true_labels_raw = [line.strip() for line in f]
        with open(predicted_labels_file, 'r') as f:
            predicted_labels_raw = [line.strip() for line in f]

        if len(true_labels_raw) != len(predicted_labels_raw):
            raise ValueError("The number of true labels does not match the number of predicted labels.")

        # Split labels and remove duplicates
        true_labels_split = [list(set(label.split(','))) for label in true_labels_raw]
        predicted_labels_split = [list(set(label.split(','))) for label in predicted_labels_raw]

        true_labels_split = [[label.strip() for label in sublist] for sublist in true_labels_split]
        predicted_labels_split = [[label.strip() for label in sublist] for sublist in predicted_labels_split]

        # Get all unique labels
        all_labels = set()
        for true_labels, predicted_labels in zip(true_labels_split, predicted_labels_split):
            all_labels.update(true_labels)
            all_labels.update(predicted_labels)
        unique_labels = sorted(list(all_labels))

        # Use MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=unique_labels)
        y_true = mlb.fit_transform(true_labels_split)
        y_pred = mlb.transform(predicted_labels_split)

        # Calculate comprehensive confusion matrix
        cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), labels=range(len(unique_labels)))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot comprehensive confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=unique_labels, yticklabels=unique_labels,
                    vmin=0, vmax=1)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Normalized Confusion Matrix (All Activities)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save confusion matrix image
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plot_path = os.path.join(os.path.dirname(output_file), 'confusion_matrix_all_activities.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate various F1 scores
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        samples_f1 = f1_score(y_true, y_pred, average='samples')
        f1_per_class = f1_score(y_true, y_pred, average=None)

        # Calculate Jaccard similarity
        jaccard_scores = []
        for true, pred in zip(true_labels_split, predicted_labels_split):
            intersection = len(set(true) & set(pred))
            union = len(set(true) | set(pred))
            jaccard_scores.append(intersection / union if union != 0 else 1.0)
        mean_jaccard = np.mean(jaccard_scores)

        # Write evaluation results
        with open(output_file, 'w') as output:
            output.write("Classification Report:\n")
            output.write(f"Micro-F1 Score (Global): {micro_f1}\n")
            output.write(f"Macro-F1 Score (Class Average): {macro_f1}\n")
            output.write(f"Weighted F1 Score (Weighted Average): {weighted_f1}\n")
            output.write(f"Sample-average F1 Score (Sample Average): {samples_f1}\n")
            output.write(f"Mean Jaccard Similarity: {mean_jaccard}\n\n")

            output.write("F1 Score per Class:\n")
            for i, label in enumerate(unique_labels):
                output.write(f"    {label}: {f1_per_class[i]}\n\n")

        print(f"Evaluation results saved to {output_file}")
        print(f"Comprehensive confusion matrix saved to {plot_path}")
        print(f"Micro-F1 Score (Global): {micro_f1:.4f}")
        print(f"Macro-F1 Score (Class Average): {macro_f1:.4f}")
        print(f"Weighted F1 Score (Weighted Average): {weighted_f1:.4f}")
        print(f"Sample-average F1 Score (Sample Average): {samples_f1:.4f}")
        print(f"Mean Jaccard Similarity: {mean_jaccard:.4f}")

    except FileNotFoundError:
        print("Error: File not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")