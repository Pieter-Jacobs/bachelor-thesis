from imports import *
from classes.Querier import Querier
from classes.DataFrameDataset import DataFrameDataset
import numpy as np


@njit
def cosine_similarity(v1, v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/((sumxx**0.5)*(sumyy**0.5))


class RankingExperiment(Querier):
    def __init__(self, model, iterator, query_function, T, Q, trainer) -> None:
        super().__init__(model=model,
                         iterator=iterator,
                         query_function=query_function,
                         T=T,
                         Q=Q,
                         )
        self.trainer = trainer

    def assign_uncertainties(self):
        self.uncertainty_matrix = []
        self.preds = np.empty((1, 14))

        self.model.eval()

        # Enable dropout
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.training = True

        # Loop through the dataset and assign the uncertainties to the examples
        with torch.no_grad():
            torch.cuda.empty_cache()
            for batch in self.iterator:
                # prediction tensors
                preds = np.array([self.model(batch.token_ids,
                                             token_type_ids=None,
                                             attention_mask=batch.mask
                                             )[0].cpu().numpy() for sample in range(int(self.T))])
                preds = np.transpose(preds, (1, 0, 2))
                preds = softmax(preds, axis=2)

                temp_preds = []
                # change the dataset, not the batches
                for pred_dist in preds:
                    avg_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    for pred in pred_dist:
                        avg_pred += pred
                    avg_pred /= len(pred_dist)
                    temp_preds.append(avg_pred.tolist())
                self.preds = np.vstack((self.preds, temp_preds))

                self.uncertainty_matrix = np.append(
                    self.uncertainty_matrix, self.query_function(preds))
        self.preds = np.delete(self.preds, 0, axis=0)

    def sort_dataset(self):
        sorted_idx = np.argsort(
            -self.uncertainty_matrix)
        self.iterator.dataset.examples = self.iterator.dataset.examples.reindex(
            sorted_idx)
        self.preds = self.preds[sorted_idx]
        # random.shuffle(self.iterator.dataset.examples)
        self.iterator.dataset.examples.index = (
            [i for i in range(len(self.iterator.dataset.examples))])

    def query(self):
        self.write_ranking(
            [example for example in self.iterator.dataset.examples.loc[0:self.Q-1]])
        labeled_examples = np.array([])
        for i in range(self.Q):
            if len(self.iterator.dataset.examples) > 0:
                self.iterator.dataset.examples.loc[i].label = self.iterator.dataset.examples.loc[i].oracle_label
                # add now labeled example to the labeled dataset
                labeled_examples = np.append(
                    labeled_examples, self.iterator.dataset.examples.loc[i])
                self.n_labeled += 1
                self.iterator.dataset.examples.drop(i, inplace=True)
        self.iterator.dataset.examples.index = (
            [i for i in range(len(self.iterator.dataset.examples))])
        return labeled_examples

    @staticmethod
    @njit
    def compute_similarities(embeddings):
        n = len(embeddings)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = cosine_similarity(
                    embeddings[i], embeddings[j])
        return distance_matrix

    def generate_one_hot(self, label):
        mapping = self.iterator.dataset.fields['label'].vocab.stoi
        one_hot = np.zeros(len(mapping))
        one_hot[0:len(one_hot)] = 0.0025
        one_hot[mapping[label]] = 0.99
        print(one_hot)
        return one_hot

    def write_ranking(self, examples):
        similarities = self.compute_similarities(
            np.array([ex.embedding for ex in examples]))
        cos_ranking = np.argsort(-similarities[0])
        labels = [ex.oracle_label for ex in examples]
        y_distributions = np.array(
            [self.generate_one_hot(label) for label in labels])
        self.uncertainty_matrix = sorted(self.uncertainty_matrix, reverse=True)
        kl_similarity_before = np.array([stats.entropy(
            self.preds[i], distribution, base=2) for i, distribution in enumerate(y_distributions)])
        df = pd.DataFrame(data={
            "label": labels,
            "cos_sim": similarities[0],
            "kl_similarity_before": kl_similarity_before,
            "uncertainty": self.uncertainty_matrix[0:self.Q]
        })
        df.to_csv("ranking.csv", mode='a')
        self.trainer.train_iterator.dataset = DataFrameDataset(
            examples=[examples[0]], fields=self.iterator.dataset.fields)
        self.trainer.training_step()
        self.assign_uncertainties()
        kl_similarity_after = np.array([stats.entropy(
            self.preds[i], distribution, base=2) for i, distribution in enumerate(y_distributions)])
        df = pd.DataFrame(data={
            "label": labels,
            "cos_sim": similarities[0],
            'kl_similarity_after': kl_similarity_after,
            "uncertainty": self.uncertainty_matrix[0:self.Q]
        })
        df.to_csv("ranking.csv", mode='a')
        print(kl_similarity_before - kl_similarity_after)
        rank_kl = np.argsort(-(kl_similarity_before - kl_similarity_after))
        f = open('ranking.txt', "a")
        f.write(str(stats.kendalltau(cos_ranking, rank_kl)[0]))
        f.write("\n")
        f.close()
