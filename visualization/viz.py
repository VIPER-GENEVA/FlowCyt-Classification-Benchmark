######### EMBEDDINGS WITH t-SNE #########
import matplotlib.pyplot as plt
import umap
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import torch

INPUT_GRAPH = 'data/sub_graph.pt' # 'data/A_graph.pt'
model = torch.load('trans/trans_gat.pt')
masked_graphs = torch.load(INPUT_GRAPH)
data22 = masked_graphs[22] #Patient23, since PyG list start from zero.
label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "black", 4: "yellow"} # 5: 'purple' if INPUT_GRAPH = 'data/A_graph.pt' since it has six classes

def visualize22(h, color):
    t_sne_embeddings = TSNE(n_components=2, perplexity=220, method='barnes_hut').fit_transform(h.detach().cpu().numpy())
    plt.scatter(t_sne_embeddings[:, 0], t_sne_embeddings[:, 1], c=color.cpu().numpy().astype(int), s=4, edgecolors='black', linewidths=0.2)

    ind1 = np.where(color.cpu().numpy() == 0)
    ind2 = np.where(color.cpu().numpy() == 1)
    ind3 = np.where(color.cpu().numpy() == 2)
    ind4 = np.where(color.cpu().numpy() == 3)
    ind5 = np.where(color.cpu().numpy() == 4)

    # Customize the legend labels
    legend_labels = ['T Lymphocytes', 'B Lymphocytes', 'Monocytes', 'Mast Cells', 'Hematopoietic']  #'Others' # Replace with your desired labels
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=label_to_color_map[class_id], markersize=5) for class_id, label in enumerate(legend_labels)]
    plt.legend(handles=legend_elements)

    plt.savefig('tsne23_100dpi.jpg', dpi=100)  # Save the image as JPG
    plt.show()

out22 = model(data22.x, data22.edge_index)
visualize22(out22, color=data22.y.cpu())  # Move color tensor to CPU

t_sne_embeddings = TSNE(n_components=2, perplexity=220, method='barnes_hut').fit_transform(out22.detach().cpu().numpy())
num_classes = 5 # 6 if INPUT_GRAPH = 'data/A_graph.pt'
fig = plt.figure(figsize=(12, 8), dpi=100)
for class_id in range(num_classes):
    plt.scatter(t_sne_embeddings[data22.y.cpu() == class_id, 0], t_sne_embeddings[data22.y.cpu() == class_id, 1], s=4, color=label_to_color_map[class_id], edgecolors='black', linewidths=0.2)

legend_labels = ['T Lymphocytes', 'B Lymphocytes', 'Monocytes', 'Mast Cells', 'Hematopoietic'] #'Others' # Replace with desired labels
legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=label_to_color_map[class_id], markersize=5) for class_id, label in enumerate(legend_labels)]
plt.legend(handles=legend_elements)
plt.savefig('TRANS23.png', dpi=100)
plt.show()

######################################## ATTENTION EXPLAINER ########################################
# AttentionExplainer uses attention coefficients to determine edge weights/opacities.
attention_explainer = Explainer(
    model=model,
    # AttentionExplainer takes an optional reduce parameter. The reduce parameter
    # allows you to set how you want to aggregate attention coefficients over layers
    # and heads. The explainer will then aggregate these values using this
    # given method to determine the edge_mask (we use the default 'max' here).
    algorithm=AttentionExplainer(),
    explanation_type='model',
    # Like PGExplainer, AttentionExplainer also does not support node_mask_type
    edge_mask_type='object',
    model_config=dict(mode='multiclass_classification', task_level='node', return_type='log_probs'),
)

data = masked_graphs[22]
node_index=10
attention_explanation = attention_explainer(data.x, data.edge_index, index=node_index)
attention_explanation.visualize_graph("attention_graph_10.png", backend="graphviz")
plt.imshow(plt.imread("attention_graph_10.png"))

################################### EXPLAINABILITY ###################################
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

data = masked_graphs[22]
node_index = 79142 # Select the node index on your choice
explanation = explainer(data.x, data.edge_index, index=node_index) 

print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path)
print(f"Subgraph visualization plot has been saved to '{path}'")

################################### DEGREES VISUALIZATION ##########################
from torch_geometric.utils import degree
import numpy as np
import matplotlib.pyplot as plt

# Get model's classifications
out = model(data.x, data.edge_index)
out = out.cpu()
data.y = data.y.cpu()
degrees = degree(data.edge_index[0]).cpu().numpy()
accuracies = []
sizes = []
# Accuracy for degrees between 0 and 5
for i in range(0, 6):
    mask = np.where(degrees == i)[0]
    accuracies.append(accuracy_score(out.argmax(dim=1)[mask], data.y[mask]))
    sizes.append(len(mask))

# Accuracy for degrees > 5
mask = np.where(degrees > 5)[0]
accuracies.append(accuracy_score(out.argmax(dim=1)[mask], data.y[mask]))
sizes.append(len(mask))

# Bar plot
fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlabel('Node degree')
ax.set_ylabel('Accuracy score')
ax.set_facecolor('#EFEEEA')
plt.bar(['0','1','2','3','4','5','>5'],
        accuracies,
        color='#0A047A')
for i in range(0, 7):
    plt.text(i, accuracies[i], f'{accuracies[i]*100:.2f}%',
             ha='center', color='#0A047A')
for i in range(0, 7):
    plt.text(i, accuracies[i]//2, sizes[i],
             ha='center', color='white')

plt.savefig('degree_accuracy.png', dpi=100)
#########################
