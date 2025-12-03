import torch
import torch.nn as nn  # nerual network
import torch.optim as optim # stocastic algo, etc
import torch.nn.functional as F # activation functions
import torchvision.transforms as transforms # transformations that can be perfromed on the dataset
import torchvision.datasets as datasets # standard dataset in pytorch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader,  Subset, random_split, ConcatDataset # easier dataset management
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os
import seaborn as sns
from itertools import product
import shutil

# need to change back later
fake_class_labelling = {
    'diffusion_GAN': 0,
    'denoising-diffusion-gan': 1,
    'stable_diffusion': 2,
    'style_gan_1': 3,
    'style_gan_2' : 4,
    'style_gan_3': 5,
    'gansformer' : 6,
    'proGAN' : 7,
    'projectedGAN' : 8,
    'taming_transformer' : 9,
}

fake_class_labelling_test = {
    'diffusion_GAN': 0,
    'denoising-diffusion-gan': 1,
    'style_gan_3': 2,
    
}
# ============ Define Model with Two Output Heads ============

class HierarchicalClassifier(nn.Module):
    def __init__(self, num_fake_classes,mode='hierarchical', backbone='resnet50', pretrained=True, dropout_rate=0.3):
        super(HierarchicalClassifier, self).__init__()

        # Load ResNet50 as the backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()  # Remove the final fully connected layer
        self.mode = mode
        self.dropout = nn.Dropout(p=dropout_rate)
        print("mode: ",self.mode)
        # Binary Classifier: Real vs. Fake
        self.binary_classifier = nn.Linear(2048, 1)  # Single output node for binary classification

        # Multiclass Classifier: Which Fake Model
        if self.mode == 'hierarchical':
          self.fake_classifier = nn.Linear(2048, num_fake_classes)  # Softmax output for fake class

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)

        real_fake_pred = torch.sigmoid(self.binary_classifier(features))  # Binary classification
        if self.mode == 'hierarchical':
          fake_class_pred = torch.softmax(self.fake_classifier(features), dim=1)  # Multiclass classification
          return real_fake_pred, fake_class_pred
        else:
          return real_fake_pred  # Only return binary classification in binary mode

# restNet50 with dropOut
class ResNet50Customed(nn.Module):
    def __init__(self, num_classes=11, dropout_rate=0.5, pretrained=True):
        super(ResNet50Customed, self).__init__()
        self.base_model = models.resnet50(pretrained=pretrained)
        num_features = self.base_model.fc.in_features
        
        # Replace the FC layer with dropout + new FC
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )  

    def forward(self, x):
        return self.base_model(x)

# ============ Custom Loss Function for Hierarchical Learning ============
class HierarchicalLoss(nn.Module):
    def __init__(self,mode='hierarchical', lambda_fake=1.0):
        
        super(HierarchicalLoss, self).__init__()
        assert mode in ['hierarchical', 'binary'], "Mode must be 'hierarchical' or 'binary'"
        self.mode = mode
        self.binary_loss = nn.BCELoss()  # Binary Cross-Entropy Loss for real vs fake
        if self.mode == 'hierarchical':
          self.fake_loss = nn.CrossEntropyLoss()  # Cross-Entropy Loss for fake class
          self.lambda_fake = lambda_fake  # Weighting factor for fake classification loss
        self.real_fake_loss = 0
        self.multi_loss = 0

    def forward(self, real_fake_pred, fake_class_pred, real_fake_labels, fake_class_labels):
        # Compute binary classification loss (applies to all images)
        real_fake_loss = self.binary_loss(real_fake_pred.squeeze(), real_fake_labels.float())
        if self.mode == 'hierarchical':
        # Compute fake classification loss (only for fake images)
          fake_mask = (real_fake_labels == 1)  # Mask where images are fake
          if fake_mask.any():
              fake_loss = self.fake_loss(fake_class_pred[fake_mask], fake_class_labels[fake_mask])
          else:
              fake_loss = torch.tensor(0.0, device=real_fake_pred.device)

          # Total loss = Binary Loss + Weighted Fake Classification Loss
          total_loss = real_fake_loss + self.lambda_fake * fake_loss
          self.real_fake_loss = real_fake_loss
          self.multi_loss = fake_loss
          return total_loss
        else:
            return real_fake_loss  # Only binary classification loss in binary mode
    def getLoss(self):
       return self.real_fake_loss.item(), self.multi_loss.item()
# define hierarchical dataset
class HierarchicalDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_labelling=None):
        """
        Expected directory structure:
          root_dir/
            real/        --> real images
            fake/        --> fake images subfolders
              GAN/
              VAE/
              Diffusion/
              ...  
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.class_labelling = class_labelling
        self.labels = []  # Each element is a tuple: (real_fake_label, fake_model_label)

        # List of fake subfolders (if any)
        fake_dir = os.path.join(root_dir, "fake")
        self.fake_classes = sorted(os.listdir(fake_dir)) if os.path.isdir(fake_dir) else []

        # Process real images (label 0 and fake label -1)
        real_dir = os.path.join(root_dir, "real")
        if os.path.isdir(real_dir):
            for fname in os.listdir(real_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(real_dir, fname))
                    self.labels.append((0, -1))  # 0 = Real; -1 means no fake model label

        # Process fake images: all subfolders in "fake"
        for fake_class in self.fake_classes:
            
            fake_class_dir = os.path.join(root_dir, "fake", fake_class)
            if not os.path.isdir(fake_class_dir):
                continue
            
            # customised labelling
            fake_class_label = self.class_labelling.get(fake_class, None) # get the value by key fake_class
            if fake_class_label is None:
                # If no manual label is provided, fall back to the index-based label
                fake_class_label = self.fake_classes.index(fake_class)

            #print("fake_class: ",fake_class)
            #print("label: ", fake_class_label)
            
            # iterate each file
            for file_in_dir in os.listdir(fake_class_dir):
                if file_in_dir.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(fake_class_dir, file_in_dir))
                    # 1 = Fake; fake_model_label = index of fake_class in sorted list or customed ordering
                    self.labels.append((1, fake_class_label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        real_fake_label, fake_model_label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Return labels as tensors
        return image, torch.tensor(real_fake_label, dtype=torch.float32), torch.tensor(fake_model_label, dtype=torch.long)

# define normalisation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
# defien data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.RandomResizedCrop(224),  # Random crop with resizing
    transforms.RandomHorizontalFlip(p=0.5),  # Flip image with 50% probability
    transforms.RandomRotation(30),  # Rotate image by ±30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color properties
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Random shifts
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# hyperParameter
num_epochs = 40

learning_rate = 5e-05
batch_size = 64 
alpha_weighting = 0.5 # for scheduler, used to calculate weighted average of performance
weight_decay = 1e-5

dropout_rate = 0.3    # for hier
#dropout_rate = 0.5   # for multiclass

# training settting
hyperTuning = False
subset = False
test_set_saved = True # flag controlling the test set saving 
#training_data = "large"
training_data = "small"
model_type = "hierarchical" # the two different classifiers
#model_type = "multi"

num_fake_classes = 10  # Change this based on how many fake models you have
mode = 'hierarchical'  # Change to 'binary' for binary classification only
model_chosen = "resnet50"

gen = torch.Generator()
gen.manual_seed(42)  # Set seed for reproducibility

# Store misclassified images, three different lists for different classifier used
misclassified_images_hier_binary , true_labels_hier_binary, pred_labels_hier_binary = [],[],[]
misclassified_images_hier_multi , true_labels_hier_multi, pred_labels_hier_multi = [],[],[]
misclassified_images_multiclass , true_labels_multiclass, pred_labels_multiclass = [],[],[]


# losses and acc for hierarchical classification
losses_and_acc_dict_hier = {
  "train_losses_bin" : [], # only use in hier
  "train_losses_multi" : [], # only use in hier
  "epoch_loss_train" : [], 
  "train_accuracies_binary" : [], # only use in hier
  "train_accuracies_fake_class" : [], 
  "val_accuracies_binary" : [], # only use in hier
  "val_accuracies_fake_class" : [] 
}
# losses and acc for multiclass classification
losses_and_acc_dict_multi = {
  "epoch_loss_train" : [],
  "train_accuracies" : [],
  "val_accuracies" : []
}
# confusion matrix, recall, precision, f1_score, auc
metric_dict = { 
  "cm_bin" : None,
  "cm_multi" : None,
  "auc_bin" : None,
  "tpr_bin" : None,
  "fpr_bin" : None,
  "thres_bin" : None,
  "auc_multi_ovr" : None,
  "auc_multi" : None,
  "tpr_multi" : None,
  "fpr_multi" : None,
  "thres_multi" : None,
  "recall_bin" : None,
  "recall_multi" : None,
  "precision_bin" : None,
  "precision_multi" : None,
  "f1_bin" : None,
  "f1_multi" : None,
  "acc_bin" : None,
  "acc_multi": None
  }

# initialise some device
# either run on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda available: ",torch.cuda.is_available())

def choose_model_type(type):
  if type == "multi":
    model = ResNet50Customed(num_classes=num_fake_classes+1, dropout_rate= dropout_rate)
    return model.to(device) , nn.CrossEntropyLoss()
  if type == "hierarchical":
    return HierarchicalClassifier(num_fake_classes,mode=mode,dropout_rate=dropout_rate ).to(device), HierarchicalLoss(lambda_fake=1.0)

# Loss and optimizer

model, criterion = choose_model_type(model_type)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# plot the loss hierarchical classifier
def plot_metrics_hierarchical(dict, metric_dict):
    #from dict to variable
    train_losses_bin = dict["train_losses_bin"]
    train_losses_multi = dict["train_losses_multi"]
    epoch_losses = dict["epoch_loss_train"]
    train_accuracies_bin = dict["train_accuracies_binary"]
    train_accuracies_multi = dict["train_accuracies_fake_class"]
    val_accuracies_bin = dict["val_accuracies_binary"]
    val_accuracies_multi = dict["val_accuracies_fake_class"]

    # from metric_dict to variable
    acc_bin = metric_dict["acc_bin"]
    acc_multi = metric_dict["acc_multi"]
    
    f1_bin = metric_dict["f1_bin"]
    f1_multi = metric_dict["f1_multi"]

    auc_bin = metric_dict["auc_bin"]
    auc_multi = metric_dict["auc_multi"]
    auc_multi_ovr = metric_dict["auc_multi_ovr"]

    recall_bin = metric_dict["recall_bin"]
    recall_multi = metric_dict["recall_multi"]

    precision_bin = metric_dict["precision_bin"]
    precision_multi = metric_dict["precision_multi"]

    cm_bin = metric_dict["cm_bin"]
    cm_multi = metric_dict["cm_multi"]

    fpr_bin = metric_dict["fpr_bin"]
    fpr_multi = metric_dict["fpr_multi"]

    tpr_bin = metric_dict["tpr_bin"]
    tpr_multi = metric_dict["tpr_multi"]

    epochs = range(1, len(epoch_losses) + 1)
    
    # printing results of the test result
    print("\n=============Real vs fake classification metrics==============")
    print(f"accuracy: {acc_bin:.4f}")
    print(f"precision: {precision_bin:.2f}")
    print(f"recall: {recall_bin:.2f}")
    print(f"f1 score: {f1_bin:.2f}")
    print(f"AUC score: {auc_bin:.2f}")
    print("\n===============================================================")

    print("\n===========AI architectures classification metrics=============")
    print(f"accuracy: {acc_multi:.4f}")
    print(f"precision: {precision_multi:.2f}")
    print(f"recall: {recall_multi:.2f}")
    print(f"f1 score: {f1_multi:.2f}")
    print(f"AUC score: {auc_multi_ovr:.2f}")
    print("\n===============================================================")

    # print results
    fig, axs = plt.subplots(2, 3, figsize=(16, 12))

    # plot losses
    axs[0,0].plot(epochs, epoch_losses, marker='o' ,linestyle='-', label="Train Loss epoch")
    axs[0,0].plot(epochs, train_losses_bin,  marker='o' ,linestyle='-', label="Train Loss true vs fake")
    axs[0,0].plot(epochs, train_losses_multi,marker='o' ,linestyle='-', label="Train Loss by Architecture")
    axs[0,0].set_xlabel("Epochs")
    axs[0,0].set_ylabel("Loss")
    axs[0,0].set_title("Loss Curve")
    axs[0,0].legend()

    # Plot Accuracy
    axs[0,1].plot(epochs, train_accuracies_bin, label="Train Accuracy true vs fake", marker='o' ,linestyle='-')
    axs[0,1].plot(epochs, val_accuracies_bin, label="Validation Accuracy true vs fake", marker='o' ,linestyle='-')
    axs[0,1].plot(epochs, train_accuracies_multi, label="Train Accuracy fake classes", marker='o' ,linestyle='-')
    axs[0,1].plot(epochs, val_accuracies_multi, label="Validation Accuracy fake classes", marker='o' ,linestyle='-')
    axs[0,1].set_xlabel("Epochs")
    axs[0,1].set_ylabel("Accuracy (%)")
    axs[0,1].set_title("Accuracy Curve")
    axs[0,1].legend()

    # Plot the confusion matrix (bin) using Seaborn
    sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1],ax=axs[0,2])
    axs[0,2].set_xlabel("Predicted Labels")
    axs[0,2].set_ylabel("True Labels")
    axs[0,2].set_title("confusion matrix real vs fake")

    # plot the confusion matrix (multi) using Seaborn
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_fake_classes), yticklabels=np.arange(num_fake_classes), ax=axs[1,0])
    axs[1,0].set_xlabel("Predicted Labels")
    axs[1,0].set_ylabel("True Labels")
    axs[1,0].set_title("confusion matrix fake classes")

    # plot the roc curve for real vs fake
    axs[1,1].plot(fpr_bin, tpr_bin, label=f' ROC curve (AUC = {auc_bin:.2f})')
    axs[1,1].plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
    axs[1,1].set_xlabel("False Positive")
    axs[1,1].set_ylabel("True Positive")
    axs[1,1].set_title("AUC graph real vs fake")
    axs[1,1].legend()

    # plot the roc curve for fake classes
    
    for i in range(num_fake_classes):
      axs[1,2].plot(fpr_multi[i], tpr_multi[i], label=f' class {i} ROC curve (AUC = {auc_multi[i]:.2f})')
    axs[1,2].plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
    axs[1,2].set_xlabel("False Positive")
    axs[1,2].set_ylabel("True Positive")
    axs[1,2].set_title("AUC graph fake classes")
    axs[1,2].legend()

    plt.savefig(f'acc and loss model={model_chosen};number of fake classes = {num_fake_classes} .png', bbox_inches='tight')
    plt.show()

# plot the loss of multiclass classifier
def plot_metrics_multi(dict, metric_dict):
    # from dict to variable
    epoch_loss_train = dict["epoch_loss_train"]
    train_accuracies = dict["train_accuracies"]
    val_accuracies = dict["val_accuracies"]

    # from metric_dict to variable
    cm_multi = metric_dict["cm_multi"]
    auc_multi = metric_dict["auc_multi"]
    tpr_multi = metric_dict["tpr_multi"]
    fpr_multi = metric_dict["fpr_multi"]
    thres_multi = metric_dict["thres_multi"]
    recall_multi = metric_dict["recall_multi"]
    precision_multi = metric_dict["precision_multi"]
    f1_multi = metric_dict["f1_multi"]

    acc_multi = metric_dict["acc_multi"]
    auc_multi_ovr = metric_dict["auc_multi_ovr"]
    # printing results of the test result

    print("===========Multiclass classification metrics=============")
    print(f"accuracy: {acc_multi:.4f}")
    print(f"precision: {precision_multi:.2f}")
    print(f"recall: {recall_multi:.2f}")
    print(f"f1 score: {f1_multi:.2f}")
    print(f"AUC score: {auc_multi_ovr:.2f}")
    print("===============================================================")

    epochs = range(1, len(epoch_loss_train) + 1)
    # print results
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # plot losses
    axs[0,0].plot(epochs, epoch_loss_train, marker='o' ,linestyle='-', label="Train Loss epoch")
    axs[0,0].set_xlabel("Epochs")
    axs[0,0].set_ylabel("Loss")
    axs[0,0].set_title("Loss Curve")
    axs[0,0].legend()

    # Plot Accuracy
    axs[0,1].plot(epochs, train_accuracies, label="Train Accuracy", marker='o' ,linestyle='-')
    axs[0,1].plot(epochs, val_accuracies, label="Validation Accuracy", marker='o' ,linestyle='-')
    axs[0,1].set_xlabel("Epochs")
    axs[0,1].set_ylabel("Accuracy (%)")
    axs[0,1].set_title("Accuracy Curve")
    axs[0,1].legend()

    # plot the confusion matrix (multi) using Seaborn
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_fake_classes+1), yticklabels=np.arange(num_fake_classes+1), ax=axs[1,0])
    axs[1,0].set_xlabel("Predicted Labels")
    axs[1,0].set_ylabel("True Labels")
    axs[1,0].set_title("confusion matrix fake classes")
    

    # plot the roc curve for fake classes
    for i in range(num_fake_classes):
      axs[1,1].plot(fpr_multi[i], tpr_multi[i], label=f' class {i} ROC curve (AUC = {auc_multi[i]:.2f})') 
    axs[1,1].plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
    axs[1,1].set_xlabel("False Positive")
    axs[1,1].set_ylabel("True Positive")
    axs[1,1].set_title("AUC graph fake classes")
    axs[1,1].legend()

    plt.savefig(f'acc and loss model={model_chosen};number of fake classes = {num_fake_classes} .png', bbox_inches='tight')
    plt.show()

# hyperparameter tuning
def hyperparam_fine_tuning(train_dataset, val_dataset ,tuning_epochs=8, mode = "hierarchical"):
  print("fine-tuning ......")
  global learning_rate, optimizer, dropout_rate, model, criterion, scheduler
  # Define grid
  learning_rates = [1e-4, 5e-5]
  optimizers = ["adam"]
  dropout_rates = [0.3,0.5]

  best_val_acc = 0
  best_config = None
  tuning_results = []

  for lr, opt_type, dr in product(learning_rates, optimizers, dropout_rates):
    print(f"\n Trying config: lr={lr}, optimizer={opt_type.upper()}, dropout={dr}")

    # Initialize model and loss
    # hier vs multi
    if mode == "hierarchical":
      model = HierarchicalClassifier(num_fake_classes, mode='hierarchical', dropout_rate=dr).to(device)
      criterion = HierarchicalLoss(mode='hierarchical', lambda_fake=1.0)
    else:
      model = ResNet50Customed(num_classes=num_fake_classes+1, dropout_rate= dropout_rate).to(device)
      criterion = nn.CrossEntropyLoss()

    if opt_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=False)

    # Reset logs
    for key in losses_and_acc_dict_hier: losses_and_acc_dict_hier[key] = []
    for key in losses_and_acc_dict_multi : losses_and_acc_dict_multi[key] = []
    for key in metric_dict: metric_dict[key] = None

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop for tuning
    if mode == "hierarchical":
      training_process(train_loader, val_loader, tuning_mode=True, epoch_num=tuning_epochs)
    else:
      training_process_multiclass(train_loader, val_loader, tuning_mode=True, epoch_num=tuning_epochs)
    # Evaluate on val
    final_acc = 0
    if mode == "hierarchical":
      val_acc = losses_and_acc_dict_hier["val_accuracies_binary"][-1]
      fake_acc = losses_and_acc_dict_hier["val_accuracies_fake_class"][-1]
      final_acc = alpha_weighting * val_acc + (1-alpha_weighting) * fake_acc
    else:
       final_acc = losses_and_acc_dict_multi["val_accuracies"][-1]

    print(f"✅ Validation accuracy (weighted avg): {final_acc:.2f}%")
    tuning_results.append((lr, opt_type, dr, final_acc))

    if final_acc > best_val_acc:
        best_val_acc = final_acc
        best_config = (lr, opt_type, dr)


  # Save best model and print results
  print("\n🏆 Best Config Found:")
  print(f"lr={best_config[0]}, batch_size={best_config[1]} (Val Acc: {best_val_acc:.2f}%)")

  # Clear logs
  for key in losses_and_acc_dict_hier: losses_and_acc_dict_hier[key] = []
  for key in losses_and_acc_dict_multi: losses_and_acc_dict_multi[key] = []
  for key in metric_dict: metric_dict[key] = None
  del misclassified_images_hier_binary[:]
  del misclassified_images_hier_multi[:]
  del true_labels_hier_binary[:]
  del pred_labels_hier_binary[:]
  del true_labels_hier_multi[:]
  del pred_labels_hier_multi[:]
  
  learning_rate, opt_type, dropout_rate = best_config
  # hier vs multi
  if mode == "hierarchical":
    model = HierarchicalClassifier(num_fake_classes, mode='hierarchical', dropout_rate=dropout_rate).to(device)
    criterion = HierarchicalLoss(mode='hierarchical', lambda_fake=1.0)
  else:
    model = ResNet50Customed(num_classes=num_fake_classes+1, dropout_rate= dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
  if opt_type == "adam":
      optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  else:
      optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

  print("\n Best Config:")
  print(f"lr={learning_rate}, optimizer={opt_type.upper()}, dropout={dropout_rate}, val_acc={best_val_acc:.2f}%")

  print("\n📊 All tried configs:")
  for (lr, opt_type, dr, val_acc) in tuning_results:
      print(f"lr={lr}, optimizer={opt_type}, dropout={dr} → val_acc: {val_acc:.2f}%")

# save the test dataset
def test_data_storage(dataset, dir, mode="hierarchical", class_label=None):
    """
    Save images from a dataset into a structured folder based on their true labels.
    - For hierarchical mode: real vs fake/[model_name]
    - For multiclass mode: [model_name]
    """
    # Clear contents inside target directory but keep the folder
    print(os.path.exists(dir))
    if os.path.exists(dir):
        for filename in os.listdir(dir):
            print("clearing: ", filename)
            file_path = os.path.join(dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"⚠️ Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(dir, exist_ok=True)

    meta_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    
    if class_label is None and hasattr(meta_dataset, 'classes'):
      class_label = {name: idx for idx, name in enumerate(meta_dataset.classes)}
    print("class_label: ", class_label)
    

    # Invert class_label if provided
    inv_class_label = {v: k for k, v in class_label.items()} if class_label else None
    print("inv_class_label: ", inv_class_label )

    for idx in tqdm(range(len(dataset)), desc="Saving images"):
        image, real_fake_label, multi_class_label = None, None, None
        if mode == "hierarchical":
          image, real_fake_label, multi_class_label = dataset[idx] 
        elif mode == "multi":
           image, multi_class_label = dataset[idx] 
        else:
            raise ValueError("Mode must be 'hierarchical' or 'multi'")

        if mode == "hierarchical":
            if real_fake_label == 0: # true label
                save_path = os.path.join(dir, "real")
            else:
                #print("multi_class_label: ", multi_class_label.item())
                class_name = inv_class_label.get(multi_class_label.item(), f"class_{multi_class_label.item()}")
                save_path = os.path.join(dir, "fake", class_name)
        elif mode == "multi":
            #print("multi_class_label: ", multi_class_label)
            class_name = inv_class_label.get(multi_class_label, f"class_{multi_class_label}")
            save_path = os.path.join(dir, class_name)
        else:
            raise ValueError("Mode must be 'hierarchical' or 'multi'")

        os.makedirs(save_path, exist_ok=True)
        image = unnormalize(image, mean, std)
        img_pil = transforms.ToPILImage()(image)
        img_pil.save(os.path.join(save_path, f"img_{idx}.png"))

    print(f"✅ Saved dataset to '{dir}' based on true labels.")

# for normal training aka not validation
def normal_training_setting(path_name, split, mode):
  def subset_procedure(subsetTrain, subset_val, subset_test,train_dataset, val_dataset,test_dataset):
    if subset:
      return subsetTrain, subset_val, subset_test
    else:
      return train_dataset, val_dataset, test_dataset
  if split:
    # choosing type of model
    if model_type == "hierarchical":
      dataset = HierarchicalDataset(root_dir=path_name,class_labelling=fake_class_labelling)  # Use custom dataset 

    elif model_type == "multi":
      # handling the folder structure for multiclass classification
      dataset = datasets.ImageFolder(root=path_name)


    print("dataset size: ", len(dataset))
    # Split dataset

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=gen)
    
    # Manually apply different transforms
    train_dataset.dataset.transform = train_transforms  # Apply augmentation only to train set
    val_dataset.dataset.transform = test_transforms
    test_dataset.dataset.transform = test_transforms  # Apply only resizing & normalization to test set
    print("train_dataset size: ", len(train_dataset))

  else: # no split required
    if model_type == "hierarchical":
      train_dataset = HierarchicalDataset(root_dir=path_name + "/train", transform=train_transforms,class_labelling=fake_class_labelling)
      test_dataset = HierarchicalDataset(root_dir=path_name + "/test", transform=test_transforms,class_labelling=fake_class_labelling)
    elif model_type == "multi":
      train_dataset = datasets.ImageFolder(root= path_name + "/train", transform=train_transforms)
      test_dataset = datasets.ImageFolder(root= path_name + "/test", transform=test_transforms)

  subset_size_train = 1400
  subset_size_val = 300
  subset_size_test = 300
  subsetTrain = Subset(train_dataset, range(subset_size_train))
  subset_val = Subset(val_dataset, range(subset_size_val))
  subset_test = Subset(test_dataset, range(subset_size_test))


  train_dataset_final, val_dataset_final, test_dataset_final = subset_procedure(subsetTrain,subset_val , subset_test,train_dataset, val_dataset,test_dataset)
  
  if hyperTuning == True:
     hyperparam_fine_tuning(train_dataset_final, val_dataset_final, mode=model_type)
  elif hyperTuning == False:
    # saving test data
    saving_directory = "testing_data/" + model_type + "/" + training_data
    if test_set_saved == False:
      if model_type == "hierarchical":
        test_data_storage(test_dataset_final, saving_directory, mode=model_type, class_label= fake_class_labelling)
      elif model_type == "multi":
        test_data_storage(test_dataset_final, saving_directory, mode=model_type)

    train_loader = DataLoader(dataset=train_dataset_final, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset_final, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset_final, batch_size=batch_size, shuffle=True)
    if model_type == "hierarchical":
      training_process(train_loader, val_loader)  
      check_accuracy(test_loader, model, "test")
    elif model_type == "multi":
      training_process_multiclass(train_loader, val_loader)
      check_accuracy_multi(test_loader, model, "test")
    else:
      print("error in mode selection")
    saving_misclassified_image(dataset=test_dataset_final)  # remember to use that line later

    # Save the model
    torch.save(model.state_dict(), '{}_{}_lr={}_epochs={}_fake-class={}_{}.pth'.format(model_chosen,model_type,learning_rate,num_epochs,num_fake_classes,training_data))

# can be used for both hier and multi
# only called when training ends (last epoch)
# nclass: fake class for hier, num of classes for multiclass classifier
def metrics_calculation(y_label, y_pred, y_score, type , dict, nclass=None, model_type = model_type):
  y_label = np.array(y_label)
  y_pred = np.array(y_pred)
  y_score = np.array(y_score)
  
  # accuracy
  acc = accuracy_score(y_label,y_pred)
  print(f"Accuracy from metrics_calculation: {acc:.4f}")

  # different setup with different type of classifiations
  if type == "bin":
    #print("bin metric")  
    dict["precision_bin"] = precision_score(y_label, y_pred, average='binary', zero_division=0)
    dict["recall_bin"] = recall_score(y_label, y_pred, average='binary')
    dict["f1_bin"] = f1_score(y_label, y_pred, average='binary')
    dict["cm_bin"] = confusion_matrix(y_label, y_pred)
    dict["acc_bin"] = acc
    
    # auc and roc
    dict["fpr_bin"] , dict["tpr_bin"], dict["thres_bin"] = roc_curve(y_label,y_score)
    dict["auc_bin"] = roc_auc_score(y_label,y_score)
    

  if type == "multiclass":
    #print("multiclass metric")
    dict["precision_multi"] = precision_score(y_label, y_pred, average='macro', zero_division=0)
    dict["recall_multi"] = recall_score(y_label, y_pred, average='macro')
    dict["f1_multi"] = f1_score(y_label, y_pred, average='macro')
    dict["cm_multi"] = confusion_matrix(y_label, y_pred)
    dict["acc_multi"] = acc
    # Multiclass AUC (One-vs-Rest for each class)
    auc = None
    
    y_label_bin = label_binarize(y_label, classes=np.arange(nclass))
    print("y_score.shape:", y_score.shape)
    print("y_label_bin.shape:", y_label_bin.shape)
    print("nclass:", nclass)
    auc = roc_auc_score(y_label_bin, y_score, multi_class='ovr')
    dict["auc_multi_ovr"] = auc
    #print(f"Multiclass AUC: {auc:.2f}")

    fpr = [[] for _ in range(nclass)]
    tpr = [[] for _ in range(nclass)]
    thresholds = [[] for _ in range(nclass)]
    auc_1 = []
    for i in range(nclass):
      fpr[i], tpr[i], thresholds[i] = roc_curve(y_label_bin[:, i], y_score[:, i])
      auc = roc_auc_score(y_label_bin[:, i], y_score[:, i])
      auc_1.append(auc)
    dict["auc_multi"] = auc_1
    dict["fpr_multi"] , dict["tpr_multi"], dict["thres_multi"] = fpr,  tpr, thresholds

# for accuracy training
def check_accuracy(loader, model, type):
    if type == "train":
      print("Checking accuracy on training data")
    elif type == "val":
      print("Checking accuracy on validation data")
    else:
      print("Checking accuracy on test data")
      
    
    num_correct_real_fake = 0
    num_correct_fake_class = 0
    num_samples = 0
    total_fake = 0

    fake_class_acc = 0
    real_fake_acc = 0
    
    # label 
    bin_label = []
    bin_pred = []
    bin_prob = []
    multi_label = []
    multi_pred = []
    multi_prob = []

    model.eval()

    with torch.no_grad(): # let pytorch know that you dont have to compute any gradient in the calculations
      for x, real_fake_labels, fake_class_labels in loader:
        # image to gpu
        x = x.to(device)
        # get the labels
        real_fake_labels = real_fake_labels.to(device)
        fake_class_labels = fake_class_labels.to(device)
        # get the prediction result
        if mode == 'hierarchical':
            real_fake_pred, fake_class_pred = model(x)
        else:
            real_fake_pred = model(x)

        # Convert real/fake prediction
        real_fake_pred_binary = (real_fake_pred > 0.5).int()
        rf_pred_squeezed = real_fake_pred_binary.squeeze()
        # Save misclassified example
        if type == "test" :
          #print("saving binary misclassified image...")     
          incorrect_mask = ~(rf_pred_squeezed == real_fake_labels)
          # getting metrics calculation (binary) here
          bin_label.extend(real_fake_labels.cpu().numpy())
          bin_pred.extend(rf_pred_squeezed.cpu().numpy())
          bin_prob.extend(real_fake_pred.cpu().numpy())
          #metrics_calculation(real_fake_labels,rf_pred_squeezed,real_fake_pred, type="bin",dict = metric_dict)
          if incorrect_mask.any():
              misclassified_images_hier_binary.append(x[incorrect_mask].cpu())
              pred_labels_hier_binary.append(rf_pred_squeezed[incorrect_mask].cpu())
              true_labels_hier_binary.append(real_fake_labels[incorrect_mask].cpu())

        # Count correct real/fake classification
        num_correct_real_fake += (rf_pred_squeezed == real_fake_labels).sum().item()
        
        # Count correct fake model classifications (only if fake)
        if mode == 'hierarchical':
            # fake_mask: a mask used for filtering fake images out of real images
            fake_mask = (real_fake_labels == 1)
            if fake_mask.any():
                fake_x = x[fake_mask]                
                _, fake_predictions = fake_class_pred[fake_mask].max(1)
                # prob of each classes from softmax
                y_prob = fake_class_pred[fake_mask]
                fake_label_filtered = fake_class_labels[fake_mask]
                # only collect misclassified iamges in final epoch and its test set
                if type == "test":
                  multi_label.extend(fake_label_filtered.cpu().numpy())
                  multi_pred.extend(fake_predictions.cpu().numpy())
                  multi_prob.extend(y_prob.cpu().numpy())
                  # getting metrics calculation (multiclass) here
                  # mask for incorrect generative model predicted
                  incorrect_model_mask = ~(fake_predictions == fake_label_filtered)
                  if incorrect_model_mask.any():
                    misclassified_images_hier_multi.append(fake_x[incorrect_model_mask].cpu())
                    pred_labels_hier_multi.append(fake_predictions[incorrect_model_mask].cpu())
                    true_labels_hier_multi.append(fake_label_filtered[incorrect_model_mask].cpu())

                num_correct_fake_class += (fake_predictions == fake_label_filtered).sum().item()
            # total_fake: total number of fake classes classified
            total_fake += (real_fake_labels == 1).sum().item()
        # num_sample: number of samples entered the classifier (both hierarchical and multiclass)
        num_samples += real_fake_labels.size(0)
        
      # calculate accuracy
      real_fake_acc = 100.0 * num_correct_real_fake / num_samples
      print(f"Real/Fake Accuracy: {real_fake_acc:.2f}%")
      if mode == 'hierarchical':
          print("total_fake: ", total_fake)
          print("num_correct_fake_class: ", num_correct_fake_class)
          fake_class_acc = 100.0 * num_correct_fake_class / max(1, total_fake)  # Avoid divide by zero
          print(f"Fake Model Classification Accuracy: {fake_class_acc:.2f}%")
          # train acc fake class 
          if type == "train":
              losses_and_acc_dict_hier["train_accuracies_fake_class"].append(fake_class_acc)
          elif type == "val":
              losses_and_acc_dict_hier["val_accuracies_fake_class"].append(fake_class_acc)
      # train acc binary result
      if type == "train":
          losses_and_acc_dict_hier["train_accuracies_binary"].append(real_fake_acc)
      elif type == "val":
          losses_and_acc_dict_hier["val_accuracies_binary"].append(real_fake_acc)
          weighted_acc = alpha_weighting * real_fake_acc + (1 - alpha_weighting) * fake_class_acc
          scheduler.step(weighted_acc / 100)

      if type == "test":
         metrics_calculation(bin_label,bin_pred,bin_prob, type="bin",dict = metric_dict, nclass=num_fake_classes, model_type = model_type)
         metrics_calculation(multi_label,multi_pred,multi_prob, type="multiclass",dict = metric_dict, nclass=num_fake_classes, model_type = model_type)
    model.train()

def check_accuracy_multi(loader, model, type):
    if type == "train":
      print("Checking accuracy on training data")
    elif type == "val":
      print("Checking accuracy on validation data")
    else:
      print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0

    multi_label = []
    multi_pred = []
    multi_prob = []

    model.eval()
    with torch.no_grad(): # let pytorch know that you dont have to compute any gradient in the calculations
      for x, y in loader:
        x = x.to(device=device)
        y = y.to(device=device)
        scores = model(x)
        _, predictions = scores.max(1) # interested in the index in the second dimension
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

        # Save misclassified example
        if type == "test" :
          # calculate matrix in last epoch of test
          multi_label.extend(y.cpu().numpy())
          multi_pred.extend(predictions.cpu().numpy())
          multi_prob.extend(scores.cpu().numpy())
          incorrect_mask = ~(predictions == y)
          if incorrect_mask.any():  
            misclassified_images_multiclass.append(x[incorrect_mask].cpu())
            true_labels_multiclass.append(predictions[incorrect_mask].cpu())
            pred_labels_multiclass.append(y[incorrect_mask].cpu())
      
      
      accuracy = float(num_correct)/float(num_samples)*100
      print(f'Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}') # converts them from tensors to flows
      # store the accuracy for graph plotting 
      if type == "train":
              losses_and_acc_dict_multi["train_accuracies"].append(accuracy)
      elif type == "val":
          losses_and_acc_dict_multi["val_accuracies"].append(accuracy)
          scheduler.step(accuracy/100)
      if type == "test":
         metrics_calculation(multi_label,multi_pred,multi_prob, type="multiclass",dict = metric_dict, nclass=num_fake_classes+1, model_type = model_type) 
         
    model.train()

def training_process_multiclass(train_loader_fn, val_loader_fn, tuning_mode = False, epoch_num = num_epochs):
   for epoch in range(epoch_num):  # epoch : 1 -> the model has seen all the images in the dataset
    running_loss = 0.0
    loop = tqdm(enumerate(train_loader_fn), total=len(train_loader_fn), leave=True)
    for batchidx, (data, targets) in loop: # data: image
      data = data.to(device=device)
      targets = targets.to(device=device)
  
      # forward
      scores = model(data)
      loss = criterion(scores, targets)

      # backward
      optimizer.zero_grad() # set the gradient to zero for each batch, clear old gradient from previous loop
      loss.backward()
        # gradient descent or adam step
      optimizer.step()
      running_loss += loss.item()

      # Update tqdm progress bar
      loop.set_description(f"Epoch [{epoch+1}/{epoch_num}]")
      loop.set_postfix(loss=loss.item())
    

    epoch_loss = running_loss / len(train_loader_fn)
    print("epoch_loss: ", epoch_loss)
    losses_and_acc_dict_multi["epoch_loss_train"].append(epoch_loss)

    check_accuracy_multi(train_loader_fn, model, "train")
    check_accuracy_multi(val_loader_fn, model, "val")

# Train Network (actual process)
def training_process(train_loader_fn, val_loader_fn, tuning_mode = False, epoch_num = num_epochs):
  #print("tuning mode: ", tuning_mode)
  #print("number of epochs: ", epoch_num)
  #print("optimiser: ", optimizer)
  for epoch in range(epoch_num):  # epoch : 1 -> the model has seen all the images in the dataset
    running_loss = 0.0
    total_bin_loss = 0.0
    total_multi_loss = 0.0
    multi_loss_batches = 0

    loop = tqdm(enumerate(train_loader_fn), total=len(train_loader_fn), leave=True)
    for batchidx, (data, real_fake_labels, fake_class_labels) in loop: # data: image
      data = data.to(device)
      real_fake_labels = real_fake_labels.to(device)
      fake_class_labels = fake_class_labels.to(device)

      optimizer.zero_grad() # set the gradient to zero for each batch, clear old gradient from previous loop
      # forward
      if mode == 'hierarchical':
        real_fake_pred, fake_class_pred = model(data)
        loss = criterion(real_fake_pred, fake_class_pred, real_fake_labels, fake_class_labels)
      else:
        real_fake_pred = model(data)
        loss = criterion(real_fake_pred, None, real_fake_labels, None)
        
      # backward
      loss.backward()
      # gradient descent or adam step
      optimizer.step()
      running_loss += loss.item()

      #print(data.shape)
      # Update tqdm progress bar
      loop.set_description(f"Epoch [{epoch+1}/{epoch_num}]")
      loop.set_postfix(loss=loss.item())
      bin_loss, multi_loss = criterion.getLoss()

      bin_loss = bin_loss.item() if isinstance(bin_loss, torch.Tensor) else bin_loss
      multi_loss = multi_loss.item() if multi_loss is not None and isinstance(multi_loss, torch.Tensor) else multi_loss

      total_bin_loss += bin_loss
      if multi_loss is not None:
        total_multi_loss += multi_loss
        multi_loss_batches += 1

    avg_bin_loss = total_bin_loss / len(train_loader_fn)
    if multi_loss_batches > 0:
        avg_multi_loss = total_multi_loss / multi_loss_batches
    else:
        avg_multi_loss = 0.0

    epoch_loss = running_loss / len(train_loader_fn)
    print("epoch_loss: ", epoch_loss)
    losses_and_acc_dict_hier["epoch_loss_train"].append(epoch_loss)
    
    print("bin, multi loss: ",avg_bin_loss, avg_multi_loss )
    losses_and_acc_dict_hier["train_losses_bin"].append(avg_bin_loss)
    losses_and_acc_dict_hier["train_losses_multi"].append(avg_multi_loss)
    
    check_accuracy(train_loader_fn, model, "train")
    check_accuracy(val_loader_fn, model, "val")

# for k-fold training
def kfold_training_setting(k_folds, cross_validation):
  k_folds = 5
  kf = KFold(n_splits=k_folds, shuffle=True)
  for fold, (train_idx, valid_idx) in enumerate(kf.split(cross_validation)):
      print(f"\nFold {fold+1}/{k_folds}")
      
      # Define Data Subsets
      train_subset = Subset(cross_validation, train_idx)
      valid_subset = Subset(cross_validation, valid_idx)
      # Manually apply different transforms
      train_subset.dataset.transform = train_transforms  # Apply augmentation only to train set
      valid_subset.dataset.transform = test_transforms  # Apply only resizing & normalization to test set
      # Define DataLoaders
      train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
      valid_loader = DataLoader(valid_subset, batch_size=64, shuffle=False)

      training_process(train_loader,valid_loader)

def find_className(pred, label, labelling_dict):
  pred_class = None
  label_class = None
  # find key by label in the customised dict
  for key,value in labelling_dict.items():
      if value == pred:
        pred_class = key
      if value == label:
        label_class = key
  return pred_class, label_class

def unnormalize(img, mean, std):
   # Ensure the mean and std are broadcastable with the image
   # std[:, None, None] reshape it from shape 3, to 3,1,1 for per-channel unnormalisation
    mean = torch.tensor(mean).to(img.device)  # Ensure mean is a tensor on the same device as img
    std = torch.tensor(std).to(img.device)    # Ensure std is a tensor on the same device as img

    # Reshape mean and std to [C, 1, 1] for broadcasting
    std_1 = std.view(3, 1, 1)  # Reshape std to [3, 1, 1]
    mean_1 = mean.view(3, 1, 1)  # Reshape mean to [3, 1, 1]

    # Unnormalize the image
    img = img * std_1 + mean_1
    return img

def clear_folder(folder_path):
    # check existence
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # Remove the file
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"{folder_path} cleared.")
    else:
        print(f"Folder {folder_path} does not exist.")

def saving_misclassified_image(dataset=None):
  # Save misclassified images to folder
  if misclassified_images_hier_binary:
      save_dir_location = f"misclassification/{model_chosen}/{model_type}_{training_data}/misclassified_bin/"
      os.makedirs(save_dir_location, exist_ok=True)
      clear_folder(save_dir_location)
      
      # torch.cat: extract the images from batches
      misclassified_imgs = torch.cat(misclassified_images_hier_binary)
      misclassified_preds = torch.cat(true_labels_hier_binary)
      misclassified_labels = torch.cat(pred_labels_hier_binary)

      # Loop through each misclassified image
      for idx in range(misclassified_imgs.size(0)):
          img = misclassified_imgs[idx]
          pred = misclassified_preds[idx].item()
          label = misclassified_labels[idx].item()

          img = unnormalize(img, mean, std)
          img_pil = transforms.ToPILImage()(img)
          # init: both real, remain unchange if pred = 0 or label = 0
          pred_class, label_class = "real","real"
          if pred == 1:
            pred_class = "fake"
          if label == 1:
            label_class = "fake"
          filename = f"img_{idx}_pred={pred_class}_label={label_class}.png"
          img_pil.save(os.path.join(save_dir_location, filename))

      print(f"Saved {misclassified_imgs.size(0)} misclassified images to '{save_dir_location}'")
  
  if misclassified_images_hier_multi:
      save_dir_location = f"misclassification/{model_chosen}/{model_type}_{training_data}/misclassified_multi/"
      os.makedirs(save_dir_location, exist_ok=True)
      clear_folder(save_dir_location)
      
      # torch.cat: extract the images from batches
      misclassified_imgs = torch.cat(misclassified_images_hier_multi)
      misclassified_preds = torch.cat(true_labels_hier_multi)
      misclassified_labels = torch.cat(pred_labels_hier_multi)

      # Loop through each misclassified image
      for idx in range(misclassified_imgs.size(0)):
          img = misclassified_imgs[idx]
          pred = misclassified_preds[idx].item()
          label = misclassified_labels[idx].item()

          img = unnormalize(img, mean, std)
          img_pil = transforms.ToPILImage()(img)
          pred_class, label_class = find_className(pred,label,fake_class_labelling)
          filename = f"img_{idx}_pred={pred_class}_label={label_class}.png"
          img_pil.save(os.path.join(save_dir_location, filename))

      print(f"Saved {misclassified_imgs.size(0)} misclassified images to '{save_dir_location}'")

  if misclassified_images_multiclass:
      save_dir_location = f"misclassification/{model_chosen}/{model_type}_{training_data}/misclassified_multi/"
      os.makedirs(save_dir_location, exist_ok=True)
      clear_folder(save_dir_location)
      
      # torch.cat: extract the images from batches
      misclassified_preds = torch.cat(true_labels_multiclass)
      misclassified_imgs = torch.cat(misclassified_images_multiclass)
      misclassified_labels = torch.cat(pred_labels_multiclass)

      # Determine class name mapping
      meta_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset

      multi_class_labelling = None
      if hasattr(meta_dataset, 'classes'):
          multi_class_labelling = {name: idx for idx, name in enumerate(meta_dataset.classes)}
      inv_class_labelling = {v: k for k, v in multi_class_labelling.items()} if multi_class_labelling else None


      # Loop through each misclassified image
      for idx in range(misclassified_imgs.size(0)):
          img = misclassified_imgs[idx]
          pred = misclassified_preds[idx].item()
          label = misclassified_labels[idx].item()

          img = unnormalize(img, mean, std)
          img_pil = transforms.ToPILImage()(img)
          pred_class = inv_class_labelling.get(pred, f"class_{pred}") if inv_class_labelling else f"class_{pred}"
          label_class = inv_class_labelling.get(label, f"class_{label}") if inv_class_labelling else f"class_{label}"

          # find the respective classname from the label mapping
          filename = f"img_{idx}_pred={pred_class}_label={label_class}.png"
          img_pil.save(os.path.join(save_dir_location, filename))

      print(f"Saved {misclassified_imgs.size(0)} misclassified images to '{save_dir_location}'")

# for ciFake
#cross_validation = datasets.ImageFolder(root= path + "/train", transform=transform)  
# for mixNmatch
#cross_validation = datasets.ImageFolder(root= "mixNmatch")

#kfold_training_setting(5, cross_validation)
if model_type == "hierarchical":
  if training_data == "small":
    normal_training_setting("mixNmatch", True, mode=model_type)
  elif training_data == "large":
    normal_training_setting("mixNmatch2", True, mode=model_type)
  else:
     ValueError("Wrong size of dataset specfied")
elif model_type == "multi":
  if training_data == "small":
    normal_training_setting("mixNmatch_multiclass", True, mode=model_type)
  elif training_data == "large":
    normal_training_setting("mixNmatch2_multiclass", True, mode=model_type)
  else:
     ValueError("Wrong size of dataset specfied")
else:
   ValueError("Wrong type of model_type")

if not hyperTuning:
  if model_type == "hierarchical":
    plot_metrics_hierarchical(losses_and_acc_dict_hier,metric_dict)
  elif model_type == "multi":
    plot_metrics_multi(losses_and_acc_dict_multi,metric_dict)
 