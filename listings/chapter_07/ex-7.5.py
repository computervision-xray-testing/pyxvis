from pyxvis.learning.transfer import generate_training_set, tfl_train
from pyxvis.learning.transfer import tfl_model, tfl_define_model, tfl_testing_accuracy
from pyxvis.io.plots import plot_confusion

# Definitions
path_dataset = '../images/objects'
nb_classes   = 4            # number of classes of the recognition problem
batch_size   = 10           # batch size in training
nb_epochs    = [40,40,40]   # epochs for Training-1, Training-2, Training-3
                            # 1st value: epochs for new layers only, 
                            # 2nd value: epochs for new and top layers of base model, 
                            # 3rd value: epochs for all layers 
                            # (eg [50,0,0], [40,50,0], etc.)
train_steps  = 10
val_steps    = 5
fc_layers    = [32, 16]     # fully connected layers after froozen layers
img_size     = [224,224]    # size of the used images
val_split    = 0.2          # portion of training set dedicated to validation, 
                            # 0 means path_dataset/val is used for validation
opti_method  = 1            # optimzer > 1: Adam, 3: SGD
base_model   = 1            # 1: MobileNet, 2: InceptionV3, 3: VGG16, 4: VGG19, 
                            # 5: ResNet50, 6: Xception, 7: MobileNetV2, 
                            # 8: DenseNet121, 9: NASNetMobile, 10: NASNetLarge
nb_layers    = -5           # layers 0... nb_layers-1 will be frozen, negative 
                            # number means the number of top layers to be unfrozen
augmentation = 0.05         # 0 : no data augmentation, otherwise it is range for 
                            # augmentation (see details in generate_training_set)

# Base model (last layer is not included removed)
bmodel      = tfl_model(base_model)
 
# New model with dense fully connected layers
model       = tfl_define_model(bmodel,fc_layers,nb_classes) 

# Training and validation sets
(train_set,
val_set)    = generate_training_set(val_split, augmentation, batch_size, 
                                    path_dataset, img_size)

# Training: Transfer learning
(model,
confusion_mtx,
acc)        = tfl_train(bmodel,model,opti_method, nb_layers,
                        train_set,train_steps,val_set,val_steps,nb_epochs,
                        path_dataset,nb_classes,img_size)

# Accuracy in testing set using best trained model
plot_confusion(confusion_mtx,acc,'Top Model: Testing in Threat Objects',0,nb_classes)
