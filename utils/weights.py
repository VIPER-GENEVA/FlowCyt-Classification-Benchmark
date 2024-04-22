import torch

def get_class_weights_tensor(graph_path):
    dataset = torch.load(graph_path)
    class_weights_tensor = []

    print(f"Number of initial graphs in the list: {len(dataset)}")
    print(f"Using graph: {graph_path}")

    for i, data in enumerate(dataset):
        num_classes = int(data.y.max()) + 1
        class_counts = torch.zeros(num_classes)
        
        for c in range(num_classes):
            class_counts[c] += (data.y == c).sum()

        total_samples = sum(class_counts)
        class_weights = total_samples / class_counts

        class_weights = class_weights / class_weights.sum()

        # OPTIONAL -- Adjust the last element of class_weights since is strongly imbalanced
        #last_weight = class_weights[-1]
        #second_last_weight = class_weights[-2]
        #class_weights[-1] = (last_weight + second_last_weight) / 2

        class_weights_tensor.append(class_weights)

    #    print(f"Graph {i + 1} Class Weights:", class_weights)
    #    print("-" * 50)

    #print(class_weights_tensor)
    print('Weights computation has been done')
    print("-" * 50)
    return class_weights_tensor