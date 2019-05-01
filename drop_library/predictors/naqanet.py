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
        "answer" key.
        """
        return self.predict_json({"passage" : passage, "question" : question})

    @overrides
    def load_line(self, line: str) -> JsonDict:
        sample = line.strip().split("\t")
        json_dict = {"passage": sample[0], "question": sample[1]}
        return json_dict

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        """
        Example Format:
            {
                "loss": 10000000.0,
                "question_id": "None",
                "answer":
                    {
                        "answer_type": "count",
                        "count": 2
                    },
                "passage_question_attention":
                    [[0.0253355223685503, 0.5861967206001282, 0.16474080085754395, 0.21797998249530792, 0.005747020710259676], [0.029191860929131508, 0.5375418663024902, 0.2956913113594055, 0.13013170659542084, 0.007443241309374571], [0.057747140526771545, 0.5951206088066101, 0.12240029871463776, 0.2180313766002655, 0.006700654048472643], [0.0697486624121666, 0.6945227384567261, 0.03133140504360199, 0.20255118608474731, 0.0018459319835528731], [0.060266438871622086, 0.46563637256622314, 0.23160749673843384, 0.23569558560848236, 0.006794052664190531], [0.0909515991806984, 0.5942181348800659, 0.07987383753061295, 0.22926151752471924, 0.0056949700228869915], [0.08830559253692627, 0.7212309837341309, 0.03848865628242493, 0.14912699162960052, 0.002847707364708185], [0.0438850075006485, 0.5492876768112183, 0.18661367893218994, 0.21424539387226105, 0.0059682042337954044], [0.025683367624878883, 0.5439653396606445, 0.3815637230873108, 0.04342569783329964, 0.005361886229366064], [0.06994843482971191, 0.542848527431488, 0.08116371929645538, 0.3013618588447571, 0.004677576012909412], [0.04958568140864372, 0.682012140750885, 0.0410219244658947, 0.22492964565753937, 0.0024505872279405594], [0.01731340028345585, 0.3680313527584076, 0.31585317850112915, 0.28249284625053406, 0.016309183090925217]],
                "question_tokens": ["How", "fruits", "are", "there", "?"],
                "passage_tokens": ["There", "is", "1", "apple", ",", "1", "banana", ",", "and", "1", "orange", "."]
            }
        NOTE: Right now it's just bare bone answer type + answer
        NOTE: We can prob do something with passage_question_attention if we were to be doing something with attention.
        """
        answer = outputs["answer"]
        answer_type = answer["answer_type"]
        answer_text = answer[answer_type]
        return "Answer type: " + answer_type + "\t Answer: " + str(answer_text) + "\n"

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text = json_dict["question"]
        passage_text = json_dict["passage"]
        return self._dataset_reader.text_to_instance(question_text, passage_text)
