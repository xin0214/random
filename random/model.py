import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

'''
Modified version of https://github.com/monologg/GoEmotions-pytorch/blob/master/model.py
'''


def get_corr(v, u, eps=1e-5):
    v_mean = torch.mean(v)
    u_mean = torch.mean(u)

    # Calculate the covariance between the vectors
    cov_vu = torch.mean((v - v_mean) * (u - u_mean))

    std_v = torch.std(v)
    std_u = torch.std(u)

    # Calculate the Pearson correlation coefficient
    corr_vu = cov_vu / ((std_v + eps) * (std_u + eps)) 
    return corr_vu

class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForMultiLabelClassificationCPCC(BertPreTrainedModel):
    def __init__(self, config, t_distance, threshold, lamda=1):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.t_distance = torch.from_numpy(t_distance)
        self.threshold = threshold
        self.lamda = lamda

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1] #[batch_size=16 default, hidden_size=768 default]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  #[batch_size=16 default, num_labels=28 default]

        # Get CPCC regularizer   
        sigmoids = self.sigmoid(logits) #[batch_size, num_labels]
        device = pooled_output.device
        label_batch = (sigmoids > self.threshold).float().to(device)
        eps = 1e-6
        label_ave = torch.matmul(label_batch.transpose(1, 0), pooled_output) 
        label_ave /= (label_batch.sum(dim=0).reshape(-1, 1) + eps) #[num_labels, hidden_size]

        num_labels = label_ave.size(0)

        # Create an empty tensor to store the L2 norm distances
        rho_dist = torch.zeros((num_labels, num_labels)).to(device)

        # Calculate the L2 norm distances between all two-row pairs
        rho_dist = torch.cdist(label_ave, label_ave, p=2)


        # Get the indices of the upper triangular part (excluding the diagonal)
        row_ind, col_ind = torch.triu_indices(num_labels, num_labels, offset=1)

        # Extract the upper triangular values
        rho_vec = rho_dist[row_ind, col_ind].view(-1)
        t_vec = self.t_distance[row_ind, col_ind].view(-1).to(device)
        
        cpcc = get_corr(rho_vec, t_vec)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels) - self.lamda * cpcc
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertMultiLevelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels_original = 28
        self.num_labels_ekman = 7
        self.num_labels_senti = 4

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_original = nn.Linear(config.hidden_size, self.num_labels_original)
        self.classifier_ekman = nn.Linear(self.num_labels_original, self.num_labels_ekman)
        self.classifier_senti = nn.Linear(self.num_labels_original, self.num_labels_senti)
        self.loss_bce = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels_original=None,
            labels_ekman=None,
            labels_senti=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        original_logits = self.classifier_original(pooled_output)
        ekman_logits = self.classifier_ekman(original_logits)
        senti_logits = self.classifier_senti(original_logits)
        outputs = (original_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if (labels_original != None and labels_ekman != None and labels_senti != None):
            loss = self.loss_bce(original_logits, labels_original) + self.loss_bce(ekman_logits, labels_ekman) + self.loss_bce(senti_logits, labels_senti) 
            outputs = (loss,) + outputs

        return outputs  # (loss), origin_logits, (hidden_states), (attentions)