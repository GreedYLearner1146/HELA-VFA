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
    loss3 = loss_NTXentLoss(classification_scores, query_labels.cuda())
    loss = loss1 + 0.5*loss2 + 0.5*loss3 
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
