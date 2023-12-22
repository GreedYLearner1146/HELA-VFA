class HELA_VFA(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(HELA_VFA, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )
        # Compute the EUCLIDEAN distance from queries to prototypes
        dists = hellinger(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores

from easyfsl.modules import resnet12

convolutional_network_output = resnet12()
convolutional_network_output.fc = nn.Flatten()
model = HELA_VFA(convolutional_network_output).cuda()
