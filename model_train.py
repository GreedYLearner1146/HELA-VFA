
################# Hyperparameters ############################################

N_TRAINING_EPISODES = 80000 # 80000,200000
N_VALIDATION_TASKS = 100

train_dataset.get_labels = lambda: [instance[1] for instance in train_dataset]
val_dataset.get_labels = lambda: [instance[1] for instance in val_dataset]

train_sampler = TaskSampler(
    train_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
)

val_sampler = TaskSampler(
    val_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
)

########## Train loader #############
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)

########## Val loader #############

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)

######################### Cross-entropy loss. #################################################
criterion1 = nn.CrossEntropyLoss()                      # Cross entropy loss as first loss.
optimizer = optim.Adam(model.parameters(), lr=0.001)    # Optimizer with learning rate of 1e-3.

############################### Model fit function ############################################

def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    classification_scores = model(
        support_images.cuda(), support_labels.cuda(), query_images.cuda()
    )

    loss1 = criterion1(classification_scores, query_labels.cuda())
    loss2 = Hesimloss(classification_scores, query_labels.cuda())
    loss = loss1 + 0.5*loss2 
    loss = loss1
    loss.backward()
    optimizer.step()

    return loss.item()

################################### Training of model ##################################

log_update_frequency = 10

all_loss = []
model.train()
with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
    for episode_index, (
        support_images,
        support_labels,
        query_images,
        query_labels,
        _,
    ) in tqdm_train:
        loss_value = fit(support_images.float(), support_labels, query_images.float(), query_labels)
        all_loss.append(loss_value)

        if episode_index % log_update_frequency == 0:
            tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))
