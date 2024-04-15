random.seed(1)   # For 1-shot learning
random.seed(10) # For 5-shot learning

# Recall we have the training and valid splits, now we do the valid and test split.

shuffled = random.sample(files_list_miniImageNet,len(files_list_miniImageNet))
trainlist_final,_ = get_training_and_valid_sets(shuffled)
_,vallist = get_training_and_valid_sets(shuffled)

# For validation and test data splitting.

def get_valid_and_test_sets(file_list):
    split = 0.50           # 20 class set as test. 
    split_index = floor(len(file_list) * split)
    # valid.
    training = file_list[:split_index]
    # test.
    validation = file_list[split_index:]
    return training, validation

validlist_final,_ = get_valid_and_test_sets(vallist)
_,testlist_final = get_valid_and_test_sets(vallist)

test_img = []

for test in testlist_final:
   data_test_img = load_images(path + '/' + test + '/')
   test_img.append(data_test_img)

############# valid, test images + labels in array list format ##################

test_img_final = []
test_label_final = []

for e in range (len(test_img)):
   for f in range (600):   # Each class has 600 images.
      test_img_final.append(test_img[e][f])
      test_label_final.append(e+80)

############# Reassemble in tuple format. ##################

test_array = []

for e,f in zip(test_img_final,test_label_final):
  test_array.append((e,f))

################## shuffle #############################

test_array = shuffle(test_array)

new_X_test = [x[0] for x in test_array]
new_y_test = [x[1] for x in test_array]

test_dataset =  miniImageNet_CustomDataset(new_X_test,new_y_test, transform=[data_transform_valtest])
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

################ Load test samplers and loaders code here. ####################################

# The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
test_dataset.get_labels = lambda: [instance[1] for instance in test_dataset]

test_sampler = TaskSampler(
    test_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

test_loader = DataLoader(
    test_dataset,
    batch_sampler=test_sampler,
    num_workers=8,  # from 12.
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

#################### Create support and query labels and images ###################

(example_support_images,
 example_support_labels,
 example_query_images,
 example_query_labels,
 example_class_ids,
) = next(iter(test_loader))

model.eval()
example_scores = model(
    example_support_images.cuda(),
    example_support_labels.cuda(),
    example_query_images.cuda(),
).detach()

_, example_predicted_labels = torch.max(example_scores.data, 1)
testlabels = [instance[1] for instance in test_dataset]

############################### First 10 classification accuracy values ##########################

Eval = []

for i in range (10):
    E = evaluate(test_loader)
    Eval.append(E)
