from typing import Any, Dict, List, Optional
import logging

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.modules import Highway
from allennlp.nn.activations import Activation
from allennlp.modules.feedforward import FeedForward
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import masked_softmax
from allennlp.training.metrics.drop_em_and_f1 import DropEmAndF1

logger = logging.getLogger(__name__)


@Model.register("naqanetic")
class NumericallyAugmentedQaNetImprovedCounting(Model):
    """
    This class augments the QANet model with some rudimentary numerical reasoning abilities, as
    published in the original DROP paper.
    The main idea here is that instead of just predicting a passage span after doing all of the
    QANet modeling stuff, we add several different "answer abilities": predicting a span from the
    question, predicting a count, or predicting an arithmetic expression.  Near the end of the
    QANet model, we have a variable that predicts what kind of answer type we need, and each branch
    has separate modeling logic to predict that answer type.  We then marginalize over all possible
    ways of getting to the right answer through each of these answer types.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 matrix_attention_layer: MatrixAttention,
                 modeling_layer: Seq2SeqEncoder,
                 dropout_prob: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 answering_abilities: List[str] = None) -> None:
        super().__init__(vocab, regularizer)


        if answering_abilities is None:
            self.answering_abilities = ["passage_span_extraction", "question_span_extraction",
                                        "addition_subtraction", "counting"]
        else:
            self.answering_abilities = answering_abilities

        text_embed_dim = text_field_embedder.get_output_dim()
        encoding_in_dim = phrase_layer.get_input_dim()
        encoding_out_dim = phrase_layer.get_output_dim()
        modeling_in_dim = modeling_layer.get_input_dim()
        modeling_out_dim = modeling_layer.get_output_dim()

        self._text_field_embedder = text_field_embedder

        self._embedding_proj_layer = torch.nn.Linear(text_embed_dim, encoding_in_dim)
        self._highway_layer = Highway(encoding_in_dim, num_highway_layers)

        self._encoding_proj_layer = torch.nn.Linear(encoding_in_dim, encoding_in_dim)
        self._phrase_layer = phrase_layer

        self._matrix_attention = matrix_attention_layer

        self._modeling_proj_layer = torch.nn.Linear(encoding_out_dim * 4, modeling_in_dim)
        self._modeling_layer = modeling_layer

        self._passage_weights_predictor = torch.nn.Linear(modeling_out_dim, 1)
        self._question_weights_predictor = torch.nn.Linear(encoding_out_dim, 1)

        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = FeedForward(modeling_out_dim + encoding_out_dim,
                                                         activations=[Activation.by_name('relu')(),
                                                                      Activation.by_name('linear')()],
                                                         hidden_dims=[modeling_out_dim,
                                                                      len(self.answering_abilities)],
                                                         num_layers=2,
                                                         dropout=dropout_prob)

        if "passage_span_extraction" in self.answering_abilities:
            self._passage_span_extraction_index = self.answering_abilities.index("passage_span_extraction")

            # takes in two encoded models, outputs start index of passage prediction (see QANet)
            self._passage_span_start_predictor = FeedForward(modeling_out_dim * 2,
                                                             activations=[Activation.by_name('relu')(),
                                                                          Activation.by_name('linear')()],
                                                             hidden_dims=[modeling_out_dim, 1],
                                                             num_layers=2)

            # takes in two encoded models, outputs end index of passage prediction (see QANet)
            self._passage_span_end_predictor = FeedForward(modeling_out_dim * 2,
                                                           activations=[Activation.by_name('relu')(),
                                                                        Activation.by_name('linear')()],
                                                           hidden_dims=[modeling_out_dim, 1],
                                                           num_layers=2)

        # doest a similar thing for the question span
        if "question_span_extraction" in self.answering_abilities:
            self._question_span_extraction_index = self.answering_abilities.index("question_span_extraction")
            self._question_span_start_predictor = FeedForward(modeling_out_dim * 2,
                                                              activations=[Activation.by_name('relu')(),
                                                                           Activation.by_name('linear')()],
                                                              hidden_dims=[modeling_out_dim, 1],
                                                              num_layers=2)
            self._question_span_end_predictor = FeedForward(modeling_out_dim * 2,
                                                            activations=[Activation.by_name('relu')(),
                                                                         Activation.by_name('linear')()],
                                                            hidden_dims=[modeling_out_dim, 1],
                                                            num_layers=2)

        # applied to each number index to output probabilities of number being 0, positive, or negative
        if "addition_subtraction" in self.answering_abilities:
            self._addition_subtraction_index = self.answering_abilities.index("addition_subtraction")
            self._number_sign_predictor = FeedForward(modeling_out_dim * 3,
                                                      activations=[Activation.by_name('relu')(),
                                                                   Activation.by_name('linear')()],
                                                      hidden_dims=[modeling_out_dim, 3],
                                                      num_layers=2)

        # simply classification of answer being a number between 0-9
        if "counting" in self.answering_abilities:
            self._counting_index = self.answering_abilities.index("counting")
            self._count_number_predictor = FeedForward(modeling_out_dim + text_embed_dim,
                                                       activations=[Activation.by_name('relu')(),
                                                                    Activation.by_name('linear')()],
                                                       hidden_dims=[modeling_out_dim, 2],
                                                       num_layers=2)

        self._drop_metrics = DropEmAndF1()
        self._dropout = torch.nn.Dropout(p=dropout_prob)
        self._mse = torch.nn.MSELoss(reduction = 'none')

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                number_indices: torch.LongTensor,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_add_sub_expressions: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()

        # embedding without dropout
        embedded_passage_wo_dropout = self._text_field_embedder(passage)

        # embedding the question and passage
        embedded_question = self._dropout(self._text_field_embedder(question))
        embedded_passage = self._dropout(embedded_passage_wo_dropout)

        # applying the highway layer to question and passage
        embedded_question = self._highway_layer(self._embedding_proj_layer(embedded_question))
        embedded_passage = self._highway_layer(self._embedding_proj_layer(embedded_passage))

        batch_size = embedded_question.size(0)

        # projecting the embeddings to be of shape (encoding_in_dim)
        projected_embedded_question = self._encoding_proj_layer(embedded_question)
        projected_embedded_passage = self._encoding_proj_layer(embedded_passage)

        # applying dropout to the vectors
        encoded_question = self._dropout(self._phrase_layer(projected_embedded_question, question_mask))
        encoded_passage = self._dropout(self._phrase_layer(projected_embedded_passage, passage_mask))


        ############################ Context-Query Attention #########################################
        # getting passage & question similarity, getting an attention matrix
        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = masked_softmax(passage_question_similarity,
                                                    question_mask,
                                                    memory_efficient=True)

        # drawing attention to which part of the passage has to do with the question??
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # drawing attention to which part of the question pertains to which part of the passage??
        # Shape: (batch_size, question_length, passage_length)
        question_passage_attention = masked_softmax(passage_question_similarity.transpose(1, 2),
                                                    passage_mask,
                                                    memory_efficient=True)

        #
        # Shape: (batch_size, passage_length, passage_length)
        passsage_attention_over_attention = torch.bmm(passage_question_attention, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_passage_vectors = util.weighted_sum(encoded_passage, passsage_attention_over_attention)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        merged_passage_attention_vectors = self._dropout(
                torch.cat([encoded_passage, passage_question_vectors,
                           encoded_passage * passage_question_vectors,
                           encoded_passage * passage_passage_vectors],
                          dim=-1))
        ########################################################################################

        ################################ Modeling Layers ##########################################
        # getting 4 of the modeled passages - 3 used for the start & end spans (M0, M1, M2), & 1 extra
        # used for the arithmetic portion (M3)
        # The recurrent modeling layers. Since these layers share the same parameters,
        # we don't construct them conditioned on answering abilities.
        modeled_passage_list = [self._modeling_proj_layer(merged_passage_attention_vectors)]
        for _ in range(4):
            modeled_passage = self._dropout(self._modeling_layer(modeled_passage_list[-1], passage_mask))
            modeled_passage_list.append(modeled_passage)
        # Pop the first one, which is input
        modeled_passage_list.pop(0)

        # The first modeling layer is used to calculate the vector representation of passage (M0??)
        passage_weights = self._passage_weights_predictor(modeled_passage_list[0]).squeeze(-1)
        passage_weights = masked_softmax(passage_weights, passage_mask)
        passage_vector = util.weighted_sum(modeled_passage_list[0], passage_weights)
        # The vector representation of question is calculated based on the unmatched encoding,
        # because we may want to infer the answer ability only based on the question words.
        question_weights = self._question_weights_predictor(encoded_question).squeeze(-1)
        question_weights = masked_softmax(question_weights, question_mask)
        question_vector = util.weighted_sum(encoded_question, question_weights)

        ########################################################################################

        # if multiple abilities (should always be the case), predict the answer type based on
        # the passage and question vector encodings (see 'Answer type prediction' in DROP paper)
        if len(self.answering_abilities) > 1:
            # Shape: (batch_size, number_of_abilities)
            answer_ability_logits = \
                self._answer_ability_predictor(torch.cat([passage_vector, question_vector], -1))
            answer_ability_log_probs = torch.nn.functional.log_softmax(answer_ability_logits, -1)
            best_answer_ability = torch.argmax(answer_ability_log_probs, 1)

        # COUNTING: pass the passage encoding through a FF layer that outputs class probabilites from 0-9,
        # gets the best number counts for
        if "counting" in self.answering_abilities:
            # # word embeddings: embedded_passage_wo_dropout, passage_mask
            # Shape: (batch_size, # of words, embedding_dim)
            embedded_words = embedded_passage_wo_dropout
            # Shape: (batch_size, # of words, embedding_dim + passage_vector_dim)
            encoded_words = torch.cat(
                [embedded_words, passage_vector.unsqueeze(1).repeat(1, embedded_words.size(1), 1)], -1)

            # Shape: (batch_size, # of words, 2)
            count_number_logits = self._count_number_predictor(encoded_words)

            # Shape: (batch_size, # of words, 2)
            count_passage_mask = passage_mask.unsqueeze(-1).repeat(1, 1, 2)
            count_number_probs = masked_softmax(count_number_logits, count_passage_mask, dim = 2, memory_efficient = True)

            # filtering out low probabilities for 1
            # count_number_probs = torch.max()

            # counting result
            # Shape: (batch_size,)
            best_count_number = torch.sum(count_number_probs[:, :, 1].squeeze(-1), dim = -1)


        if "passage_span_extraction" in self.answering_abilities:
            # Shape: (batch_size, passage_length, modeling_dim * 2))
            passage_for_span_start = torch.cat([modeled_passage_list[0], modeled_passage_list[1]], dim=-1)
            # Shape: (batch_size, passage_length)
            passage_span_start_logits = self._passage_span_start_predictor(passage_for_span_start).squeeze(-1)
            # Shape: (batch_size, passage_length, modeling_dim * 2)
            passage_for_span_end = torch.cat([modeled_passage_list[0], modeled_passage_list[2]], dim=-1)
            # Shape: (batch_size, passage_length)
            passage_span_end_logits = self._passage_span_end_predictor(passage_for_span_end).squeeze(-1)
            # Shape: (batch_size, passage_length)
            passage_span_start_log_probs = util.masked_log_softmax(passage_span_start_logits, passage_mask)
            passage_span_end_log_probs = util.masked_log_softmax(passage_span_end_logits, passage_mask)

            # Info about the best passage span prediction
            passage_span_start_logits = util.replace_masked_values(passage_span_start_logits, passage_mask, -1e7)
            passage_span_end_logits = util.replace_masked_values(passage_span_end_logits, passage_mask, -1e7)
            # Shape: (batch_size, 2)
            best_passage_span = get_best_span(passage_span_start_logits, passage_span_end_logits)
            # Shape: (batch_size, 2)
            best_passage_start_log_probs = \
                torch.gather(passage_span_start_log_probs, 1, best_passage_span[:, 0].unsqueeze(-1)).squeeze(-1)
            best_passage_end_log_probs = \
                torch.gather(passage_span_end_log_probs, 1, best_passage_span[:, 1].unsqueeze(-1)).squeeze(-1)
            # Shape: (batch_size,)
            best_passage_span_log_prob = best_passage_start_log_probs + best_passage_end_log_probs
            if len(self.answering_abilities) > 1:
                best_passage_span_log_prob += answer_ability_log_probs[:, self._passage_span_extraction_index]

        if "question_span_extraction" in self.answering_abilities:
            # Shape: (batch_size, question_length)
            encoded_question_for_span_prediction = \
                torch.cat([encoded_question,
                           passage_vector.unsqueeze(1).repeat(1, encoded_question.size(1), 1)], -1)
            question_span_start_logits = \
                self._question_span_start_predictor(encoded_question_for_span_prediction).squeeze(-1)
            # Shape: (batch_size, question_length)
            question_span_end_logits = \
                self._question_span_end_predictor(encoded_question_for_span_prediction).squeeze(-1)
            question_span_start_log_probs = util.masked_log_softmax(question_span_start_logits, question_mask)
            question_span_end_log_probs = util.masked_log_softmax(question_span_end_logits, question_mask)

            # Info about the best question span prediction
            question_span_start_logits = \
                util.replace_masked_values(question_span_start_logits, question_mask, -1e7)
            question_span_end_logits = \
                util.replace_masked_values(question_span_end_logits, question_mask, -1e7)
            # Shape: (batch_size, 2)
            best_question_span = get_best_span(question_span_start_logits, question_span_end_logits)
            # Shape: (batch_size, 2)
            best_question_start_log_probs = \
                torch.gather(question_span_start_log_probs, 1, best_question_span[:, 0].unsqueeze(-1)).squeeze(-1)
            best_question_end_log_probs = \
                torch.gather(question_span_end_log_probs, 1, best_question_span[:, 1].unsqueeze(-1)).squeeze(-1)
            # Shape: (batch_size,)
            best_question_span_log_prob = best_question_start_log_probs + best_question_end_log_probs
            if len(self.answering_abilities) > 1:
                best_question_span_log_prob += answer_ability_log_probs[:, self._question_span_extraction_index]

        if "addition_subtraction" in self.answering_abilities:
            # Shape: (batch_size, # of numbers in the passage)
            # [5, 10, -1]
            number_indices = number_indices.squeeze(-1)

            # the last element in number_indices is always -1 (artificial)
            # [1, 1, 0]
            number_mask = (number_indices != -1).long()
            # [5, 10, 0]
            clamped_number_indices = util.replace_masked_values(number_indices, number_mask, 0)

            # M0 and M3
            # Maybe Shape: (batch_size, 2 * modeled_passage_output_dim)
            encoded_passage_for_numbers = torch.cat([modeled_passage_list[0], modeled_passage_list[3]], dim=-1)

            # Shape: (batch_size, # of numbers in the passage, encoding_dim)
            encoded_numbers = torch.gather(
                    encoded_passage_for_numbers,
                    1,
                    clamped_number_indices.unsqueeze(-1).expand(-1, -1, encoded_passage_for_numbers.size(-1)))

            # Shape: (batch_size, # of numbers in the passage)
            # Maybe Shape: (batch_size, # of numbers in the passage, 3 * modeled_passage_output_dim)
            encoded_numbers = torch.cat(
                    [encoded_numbers, passage_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)

            # Shape: (batch_size, # of numbers in the passage, 3)
            number_sign_logits = self._number_sign_predictor(encoded_numbers)
            # The log probabilities of each sign for each number
            # Shape: (batch_size, # of numbers in the passage, 3)
            number_sign_log_probs = torch.nn.functional.log_softmax(number_sign_logits, -1)

            # use best_signs_for_numbers to get the answer
            # Shape: (batch_size, # of numbers in passage).
            best_signs_for_numbers = torch.argmax(number_sign_log_probs, -1)
            # For padding numbers, the best sign masked as 0 (not included).
            best_signs_for_numbers = util.replace_masked_values(best_signs_for_numbers, number_mask, 0)

            ########################### for log prob ###########################
            # Shape: (batch_size, # of numbers in passage)
            best_signs_log_probs = torch.gather(
                    number_sign_log_probs, 2, best_signs_for_numbers.unsqueeze(-1)).squeeze(-1)
            # the probs of the masked positions should be 1 so that it will not affect the joint probability
            # TODO: this is not quite right, since if there are many numbers in the passage,
            # TODO: the joint probability would be very small.
            best_signs_log_probs = util.replace_masked_values(best_signs_log_probs, number_mask, 0)
            # Shape: (batch_size,)
            best_combination_log_prob = best_signs_log_probs.sum(-1)
            if len(self.answering_abilities) > 1:
                best_combination_log_prob += answer_ability_log_probs[:, self._addition_subtraction_index]

        output_dict = {}

        # If answer is given, compute the loss.
        if answer_as_passage_spans is not None or answer_as_question_spans is not None \
                or answer_as_add_sub_expressions is not None or answer_as_counts is not None:

            log_marginal_likelihood_list = []

            for answering_ability in self.answering_abilities:
                if answering_ability == "passage_span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_passage_span_starts = answer_as_passage_spans[:, :, 0]
                    gold_passage_span_ends = answer_as_passage_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_passage_span_mask = (gold_passage_span_starts != -1).long()
                    clamped_gold_passage_span_starts = \
                        util.replace_masked_values(gold_passage_span_starts, gold_passage_span_mask, 0)
                    clamped_gold_passage_span_ends = \
                        util.replace_masked_values(gold_passage_span_ends, gold_passage_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_span_starts = \
                        torch.gather(passage_span_start_log_probs, 1, clamped_gold_passage_span_starts)
                    log_likelihood_for_passage_span_ends = \
                        torch.gather(passage_span_end_log_probs, 1, clamped_gold_passage_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_spans = \
                        log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_passage_spans = \
                        util.replace_masked_values(log_likelihood_for_passage_spans, gold_passage_span_mask, -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_passage_span = util.logsumexp(log_likelihood_for_passage_spans)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)

                elif answering_ability == "question_span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_question_span_starts = answer_as_question_spans[:, :, 0]
                    gold_question_span_ends = answer_as_question_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_question_span_mask = (gold_question_span_starts != -1).long()
                    clamped_gold_question_span_starts = \
                        util.replace_masked_values(gold_question_span_starts, gold_question_span_mask, 0)
                    clamped_gold_question_span_ends = \
                        util.replace_masked_values(gold_question_span_ends, gold_question_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_span_starts = \
                        torch.gather(question_span_start_log_probs, 1, clamped_gold_question_span_starts)
                    log_likelihood_for_question_span_ends = \
                        torch.gather(question_span_end_log_probs, 1, clamped_gold_question_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_spans = \
                        log_likelihood_for_question_span_starts + log_likelihood_for_question_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_question_spans = \
                        util.replace_masked_values(log_likelihood_for_question_spans,
                                                   gold_question_span_mask,
                                                   -1e7)
                    # Shape: (batch_size, )
                    # pylint: disable=invalid-name
                    log_marginal_likelihood_for_question_span = \
                        util.logsumexp(log_likelihood_for_question_spans)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span)

                elif answering_ability == "addition_subtraction":
                    # The padded add-sub combinations use 0 as the signs for all numbers, and we mask them here.
                    # Shape: (batch_size, # of combinations)
                    gold_add_sub_mask = (answer_as_add_sub_expressions.sum(-1) > 0).float()
                    # Shape: (batch_size, # of numbers in the passage, # of combinations)
                    gold_add_sub_signs = answer_as_add_sub_expressions.transpose(1, 2)
                    # Shape: (batch_size, # of numbers in the passage, # of combinations)
                    log_likelihood_for_number_signs = torch.gather(number_sign_log_probs, 2, gold_add_sub_signs)
                    # the log likelihood of the masked positions should be 0
                    # so that it will not affect the joint probability
                    log_likelihood_for_number_signs = \
                        util.replace_masked_values(log_likelihood_for_number_signs, number_mask.unsqueeze(-1), 0)
                    # Shape: (batch_size, # of combinations)
                    log_likelihood_for_add_subs = log_likelihood_for_number_signs.sum(1)
                    # For those padded combinations, we set their log probabilities to be very small negative value
                    log_likelihood_for_add_subs = \
                        util.replace_masked_values(log_likelihood_for_add_subs, gold_add_sub_mask, -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_add_sub = util.logsumexp(log_likelihood_for_add_subs)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_add_sub)

                elif answering_ability == "counting":
                    # Count answers are padded with label -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    # Shape: (batch_size, # of count answers)
                    gold_count_mask = (answer_as_counts != -1).long()
                    # # Shape: (batch_size, # of count answers)
                    clamped_gold_counts = util.replace_masked_values(answer_as_counts, gold_count_mask, 0)

                    # log_likelihood_for_counts = torch.gather(count_number_log_probs, 1, clamped_gold_counts)
                    # # For those padded spans, we set their log probabilities to be very small negative value
                    # log_likelihood_for_counts = \
                    #     util.replace_masked_values(log_likelihood_for_counts, gold_count_mask, -1e7)
                    # # Shape: (batch_size, )
                    # log_marginal_likelihood_for_count = util.logsumexp(log_likelihood_for_counts)

                    # Shape: (batch_size, # of count answers)
                    repeated_best_count_number = best_count_number.unsqueeze(-1).repeat(1, clamped_gold_counts.shape[-1])
                    # Shape: (batch_size,)
                    count_mse_loss = torch.mean(self._mse(repeated_best_count_number, clamped_gold_counts.float()), dim = -1)

                    # negative because it negates later
                    log_marginal_likelihood_list.append(-count_mse_loss)

                    logger.info("MSE")
                    logger.info(count_mse_loss)
                    logger.info("Predicted")
                    logger.info(best_count_number)
                    logger.info("Expected")
                    logger.info(clamped_gold_counts)
                   

                else:
                    raise ValueError(f"Unsupported answering ability: {answering_ability}")

            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]

            # logger.info(log_marginal_likelihood_list)
            output_dict["loss"] = - marginal_log_likelihood.mean()

        # Compute the metrics and add the tokenized input to the output.
        if metadata is not None:
            output_dict["question_id"] = []
            output_dict["answer"] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])

                if len(self.answering_abilities) > 1:
                    predicted_ability_str = self.answering_abilities[best_answer_ability[i].detach().cpu().numpy()]
                else:
                    predicted_ability_str = self.answering_abilities[0]

                answer_json: Dict[str, Any] = {}

                # We did not consider multi-mention answers here
                if predicted_ability_str == "passage_span_extraction":
                    answer_json["answer_type"] = "passage_span"
                    passage_str = metadata[i]['original_passage']
                    offsets = metadata[i]['passage_token_offsets']
                    predicted_span = tuple(best_passage_span[i].detach().cpu().numpy())
                    start_offset = offsets[predicted_span[0]][0]
                    end_offset = offsets[predicted_span[1]][1]
                    predicted_answer = passage_str[start_offset:end_offset]
                    answer_json["value"] = predicted_answer
                    answer_json["spans"] = [(start_offset, end_offset)]
                elif predicted_ability_str == "question_span_extraction":
                    answer_json["answer_type"] = "question_span"
                    question_str = metadata[i]['original_question']
                    offsets = metadata[i]['question_token_offsets']
                    predicted_span = tuple(best_question_span[i].detach().cpu().numpy())
                    start_offset = offsets[predicted_span[0]][0]
                    end_offset = offsets[predicted_span[1]][1]
                    predicted_answer = question_str[start_offset:end_offset]
                    answer_json["value"] = predicted_answer
                    answer_json["spans"] = [(start_offset, end_offset)]
                elif predicted_ability_str == "addition_subtraction":  # plus_minus combination answer
                    # calculate answer
                    answer_json["answer_type"] = "arithmetic"
                    original_numbers = metadata[i]['original_numbers']
                    sign_remap = {0: 0, 1: 1, 2: -1}
                    predicted_signs = [sign_remap[it] for it in best_signs_for_numbers[i].detach().cpu().numpy()]
                    result = sum([sign * number for sign, number in zip(predicted_signs, original_numbers)])
                    predicted_answer = str(result)
                    offsets = metadata[i]['passage_token_offsets']
                    number_indices = metadata[i]['number_indices']
                    number_positions = [offsets[index] for index in number_indices]
                    answer_json['numbers'] = []
                    for offset, value, sign in zip(number_positions, original_numbers, predicted_signs):
                        answer_json['numbers'].append({'span': offset, 'value': value, 'sign': sign})
                    if number_indices[-1] == -1:
                        # There is a dummy 0 number at position -1 added in some cases; we are
                        # removing that here.
                        answer_json["numbers"].pop()
                    answer_json["value"] = result
                elif predicted_ability_str == "counting":
                    answer_json["answer_type"] = "count"
                    predicted_count = best_count_number[i].detach().cpu().numpy()
                    predicted_answer = str(predicted_count)
                    answer_json["count"] = predicted_count
                else:
                    raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")

                output_dict["question_id"].append(metadata[i]["question_id"])
                output_dict["answer"].append(answer_json)
                answer_annotations = metadata[i].get('answer_annotations', [])
                if answer_annotations:
                    self._drop_metrics(predicted_answer, answer_annotations)
            # This is used for the demo.
            output_dict["passage_question_attention"] = passage_question_attention
            output_dict["question_tokens"] = question_tokens
            output_dict["passage_tokens"] = passage_tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}
