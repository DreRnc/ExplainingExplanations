from sentence_transformers.util import cos_sim
import numpy as np
from datasets import Metric

class SbertMetric(Metric):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def _compute(self, predictions, references):
		pred_emb = self.model.encode(predictions)
		label_emb = self.model.encode(references)

		similarities = [cos_sim(pred, label) for pred, label in zip(pred_emb, label_emb)]
		average_similarity = np.mean(similarities)

		output = {'average_similarity': average_similarity}
		return output