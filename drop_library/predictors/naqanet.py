from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('naqanet')
class NaqanetPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.reading_comprehension.NumericallyAugmentedQaNet` model.
    """

    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.
        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"passage" : passage, "question" : question})

    @overrides
    def load_line(self, line: str) -> JsonDict:
        sample = line.strip().split("\t")
        json_dict = {"passage": sample[0], "question": sample[1]}
        return json_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text = json_dict["question"]
        passage_text = json_dict["passage"]
        return self._dataset_reader.text_to_instance(question_text, passage_text)
