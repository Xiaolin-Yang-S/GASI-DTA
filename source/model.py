import torch
from torch.nn import Linear, ReLU, Dropout, LSTM
from torch_geometric.nn import GCNConv, global_mean_pool as gep


vector_operations = {
    "cat": (lambda x, y: torch.cat((x, y), -1), lambda dim: 2 * dim),
    "add": (torch.add, lambda dim: dim),
    "sub": (torch.sub, lambda dim: dim),
    "mul": (torch.mul, lambda dim: dim),
    "combination1": (lambda x, y: torch.cat((x, y, torch.add(x, y)), -1), lambda dim: 3 * dim),
    "combination2": (lambda x, y: torch.cat((x, y, torch.sub(x, y)), -1), lambda dim: 3 * dim),
    "combination3": (lambda x, y: torch.cat((x, y, torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination4": (lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 2 * dim),
    "combination5": (lambda x, y: torch.cat((torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination6": (lambda x, y: torch.cat((torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination7": (
    lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination8": (lambda x, y: torch.cat((x, y, torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination9": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination10": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 4 * dim),
    "combination11": (
    lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 5 * dim)
}


class LinearBlock(torch.nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = [x]
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)
        return embeddings


class ResLinearBlock(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=8, dropout_rate=0.2):
        super(ResLinearBlock, self).__init__()

        self.layer1 = Linear(input_dim, hidden_dim)
        self.layer2 = Linear(hidden_dim, output_dim)
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            layer = Linear(hidden_dim, hidden_dim)
            self.layers.append(layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = [0, 1, 2, 3, 4, 5, 6, 7]
        self.dropout_layers_index = [0, 1, 2, 3, 4, 5, 6, 7]

    def forward(self, x):

        output = self.dropout(self.relu(self.layer1(x)))
        last_output = output
        for layer_index in range(len(self.layers)):
            if layer_index != 0 and layer_index % 2 == 0:
                output = output + last_output
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            if layer_index != 0 and layer_index % 2 == 0:
                last_output = output

        output = self.dropout(self.layer2(output))

        return output


class GCNBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[],
                 supplement_mode=None):
        super(GCNBlock, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                conv_layer_input = supplement_dim_func(gcn_layers_dim[i])
            else:
                conv_layer_input = gcn_layers_dim[i]
            conv_layer = GCNConv(conv_layer_input, gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch, supplement_x=None):
        output = x
        embeddings = [x]

        for conv_layer_index in range(len(self.conv_layers)):
            if supplement_x is not None and conv_layer_index == 1:
                output = self.supplement_func(output, supplement_x)

            output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(gep(output, batch))
        return embeddings


class GCNModel(torch.nn.Module):
    def __init__(self, layers_dim, supplement_mode=None):
        super(GCNModel, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GCNBlock(layers_dim, relu_layers_index=range(self.num_layers),
                                   supplement_mode=supplement_mode)

    def forward(self, graph_batchs, supplement_x=None):

        if supplement_x is not None:
            supplement_i = 0
            for graph_batch in graph_batchs:
                graph_batch.__setitem__('supplement_x',
                                        supplement_x[supplement_i: supplement_i + graph_batch.num_graphs])
                supplement_i += graph_batch.num_graphs

            embedding_batchs = list(map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch,
                                                                      supplement_x=graph.supplement_x[
                                                                          graph.batch.int().cpu().numpy()]),
                                        graph_batchs))
        else:
            embedding_batchs = list(
                map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))

        embeddings = []
        for i in range(self.num_layers + 1):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings


class GASIDTA(torch.nn.Module):
    def __init__(self, mg_init_dim=78, pg_init_dim=54,  embedding_dim=128):
        super(GASIDTA, self).__init__()

        self.drug_LSTMNet = LSTM(input_size=384, hidden_size=128, num_layers=3, bidirectional=True, batch_first=True,
                                 dropout=0.2)
        self.target_LSTMNet = LSTM(input_size=768, hidden_size=128, num_layers=3, bidirectional=True, batch_first=True,
                                   dropout=0.2)

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4]

        drug_output_dims = [drug_graph_dims[-1] + 256, 1024, embedding_dim]
        target_output_dims = [target_graph_dims[-1] + 256, 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.drug_graph_conv = GCNModel(drug_graph_dims)
        self.target_graph_conv = GCNModel(target_graph_dims)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],
                                                dropout_layers_index=[0, 1])

    def forward(self, drug_graph_batchs, target_graph_batchs):

        drug_seq_embedding, (dhn, dcn) = self.drug_LSTMNet(drug_graph_batchs[0].seq_x)
        target_seq_embedding, (tdn, tcn) = self.target_LSTMNet(target_graph_batchs[0].seq_x)

        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]

        drug_embedding = torch.cat((drug_seq_embedding, drug_graph_embedding), 1)
        target_embedding = torch.cat((target_seq_embedding, target_graph_embedding), 1)

        drug_output_embedding = self.drug_output_linear(drug_embedding)[-1]
        target_output_embedding = self.target_output_linear(target_embedding)[-1]

        return drug_output_embedding, target_output_embedding


class GASIDTA_cold(torch.nn.Module):
    def __init__(self, mg_init_dim=78, pg_init_dim=54, embedding_dim=128):
        super(GASIDTA_cold, self).__init__()
        print('DAS')

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4]

        drug_output_dims = [drug_graph_dims[-1] + 256, 1024, embedding_dim]
        target_output_dims = [target_graph_dims[-1] + 256, 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.drug_graph_conv = GCNModel(drug_graph_dims)
        self.target_graph_conv = GCNModel(target_graph_dims)

        self.drug_seq_linear = ResLinearBlock(384, 256, 256, dropout_rate= 0.6)
        self.target_seq_linear = ResLinearBlock(768, 256, 256, dropout_rate= 0.6)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],
                                                dropout_layers_index=[0, 1])

    def forward(self, drug_graph_batchs, target_graph_batchs):

        drug_seq_embedding = self.drug_seq_linear(drug_graph_batchs[0].seq_x)
        target_seq_embedding = self.target_seq_linear(target_graph_batchs[0].seq_x)

        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]

        drug_embedding = torch.cat((drug_seq_embedding, drug_graph_embedding), 1)
        target_embedding = torch.cat((target_seq_embedding, target_graph_embedding), 1)

        drug_output_embedding = self.drug_output_linear(drug_embedding)[-1]
        target_output_embedding = self.target_output_linear(target_embedding)[-1]

        return drug_output_embedding, target_output_embedding


class GASIDTA_cold_drug(torch.nn.Module):
    def __init__(self, mg_init_dim=78, pg_init_dim=54, embedding_dim=128):
        super(GASIDTA_cold_drug, self).__init__()

        self.target_LSTMNet = LSTM(input_size=768, hidden_size=128, num_layers=3, bidirectional=True, batch_first=True,
                                   dropout=0.2)

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4]

        drug_output_dims = [drug_graph_dims[-1] + 256, 1024, embedding_dim]
        target_output_dims = [target_graph_dims[-1] + 256, 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.drug_graph_conv = GCNModel(drug_graph_dims)
        self.target_graph_conv = GCNModel(target_graph_dims)

        self.drug_seq_linear = ResLinearBlock(384, 256, 256, dropout_rate=0.5)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],
                                                dropout_layers_index=[0, 1])

    def forward(self, drug_graph_batchs, target_graph_batchs):
        drug_seq_embedding = self.drug_seq_linear(drug_graph_batchs[0].seq_x)
        target_seq_embedding, (tdn, tcn) = self.target_LSTMNet(target_graph_batchs[0].seq_x)

        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]

        # 融合策略
        drug_embedding = torch.cat((drug_seq_embedding, drug_graph_embedding), 1)
        target_embedding = torch.cat((target_seq_embedding, target_graph_embedding), 1)

        drug_output_embedding = self.drug_output_linear(drug_embedding)[-1]
        target_output_embedding = self.target_output_linear(target_embedding)[-1]

        return drug_output_embedding, target_output_embedding


class GASIDTA_cold_target(torch.nn.Module):
    def __init__(self, mg_init_dim=78, pg_init_dim=54, embedding_dim=128):
        super(GASIDTA_cold_target, self).__init__()
        print('DAS')

        self.drug_LSTMNet = LSTM(input_size=384, hidden_size=128, num_layers=3, bidirectional=True, batch_first=True,
                                 dropout=0.2)

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4]

        drug_output_dims = [drug_graph_dims[-1] + 256, 1024, embedding_dim]
        target_output_dims = [target_graph_dims[-1] + 256, 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.drug_graph_conv = GCNModel(drug_graph_dims)
        self.target_graph_conv = GCNModel(target_graph_dims)


        self.target_seq_linear = ResLinearBlock(768, 256, 256, dropout_rate=0.3)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],
                                                dropout_layers_index=[0, 1])

    def forward(self, drug_graph_batchs, target_graph_batchs):

        drug_seq_embedding, (dhn, dcn) = self.drug_LSTMNet(drug_graph_batchs[0].seq_x)
        target_seq_embedding = self.target_seq_linear(target_graph_batchs[0].seq_x)

        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]


        drug_embedding = torch.cat((drug_seq_embedding, drug_graph_embedding), 1)
        target_embedding = torch.cat((target_seq_embedding, target_graph_embedding), 1)

        drug_output_embedding = self.drug_output_linear(drug_embedding)[-1]
        target_output_embedding = self.target_output_linear(target_embedding)[-1]

        return drug_output_embedding, target_output_embedding


class Predictor(torch.nn.Module):
    def __init__(self, embedding_dim=128, output_dim=1, prediction_mode="cat"):
        super(Predictor, self).__init__()
        print('Predictor Loaded')

        self.prediction_func, prediction_dim_func = vector_operations[prediction_mode]
        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):
        drug_id, target_id, y = data.drug_id, data.target_id, data.y

        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]

        concat_feature = self.prediction_func(drug_feature, target_feature)

        mlp_embeddings = self.mlp(concat_feature)
        link_embeddings = mlp_embeddings[-2]
        out = mlp_embeddings[-1]

        return out, link_embeddings
