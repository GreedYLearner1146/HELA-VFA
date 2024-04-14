files_list_miniImageNet = []

for filename in sorted(os.listdir(path),key=natural_sort_key):
    files_list_miniImageNet.append(filename)

shuffled = random.sample(files_list_miniImageNet,len(files_list_miniImageNet))

def fine_tune_datasets(file_list):
    split = 0.80
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    validation = file_list[split_index:]
    return training, validation

_, FTlist_final = fine_tune_datasets(shuffled)

################################################################################

FT = []

for ft in FTlist_final:
   FT_Img = load_images(path + '/' + ft + '/')
   FT.append(FT_Img)

FT_img = []
FT_label = []

for g in range (len(FT)):
  for h in range (600):
      FT_img.append(FT[g][h])
      FT_label.append(g)

ft_array = []

for g,h in zip(FT_img,FT_label):
  ft_array.append((g,h))

FT_array = shuffle(ft_array)

new_X_FT = [x[0] for x in FT_array]
new_y_FT = [x[1] for x in FT_array]

################################################################################

FT_dataset =  miniImageNet_CustomDataset(new_X_FT, new_y_FT, transform=data_transform_test)
FT_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

FT_dataset.get_labels = lambda: [instance[1] for instance in FT_dataset]

FT_sampler = TaskSampler(
    FT_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

FT_loader = DataLoader(
    FT_dataset,
    batch_sampler=FT_sampler,
    num_workers=8,  # from 12.
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

############################################################################################

(example_support_images,
 example_support_labels,
 example_query_images,
 example_query_labels,
 example_class_ids,
) = next(iter(FT_loader))

model.eval()
example_scores = model(
    example_support_images.cuda(),
    example_support_labels.cuda(),
    example_query_images.cuda(),
).detach()

_, example_predicted_labels = torch.max(example_scores.data, 1)
FTlabels = [instance[1] for instance in FT_dataset]

############################################################################################
# For seeing the fine-tuned evaluation results.
Eval = []

for i in range (10):
    E = evaluate(FT_loader)
    Eval.append(E)
