from torchtext import data

class TrecDataset(data.TabularDataset):
    dirname = 'data'
    @classmethod

    def splits(cls, question_id, question_field, question_pos, question_dep, head_question, head_question_pos,
               head_question_dep, answer_field, answer_pos, answer_dep, head_answer, head_answer_pos, head_answer_dep,
               external_field, label_field, idf_question, idf_answer, question_is_num, answer_is_num, question_ner,
               answer_ner, root='.data',
               train='trecqa.train.tsv', validation='trecqa.dev.tsv', test='trecqa.test.tsv'):
        path = './data'
        return super(TrecDataset, cls).splits(
            path, root, train, validation, test,
            format='TSV', fields=[('qid', question_id), ('label', label_field), ('question', question_field),
                                  ('answer', answer_field), ('ext_feat', external_field),
                                  ('head_question', head_question), ('head_q_pos', head_question_pos),
                                  ('head_q_dep', head_question_dep), ('question_word_pos',question_pos),
                                  ('question_word_dep', question_dep),
                                  ('head_answer', head_answer), ('head_a_pos', head_answer_pos),
                                  ('head_a_dep', head_answer_dep), ('answer_word_pos',answer_pos),
                                  ('answer_word_dep', answer_dep), ('question_idf', idf_question),
                                  ('answer_idf', idf_answer), ('question_is_num', question_is_num),
                                  ('answer_is_num', answer_is_num), ('question_ner', question_ner),
                                  ('answer_ner', answer_ner)])
