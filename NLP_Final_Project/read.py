from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer


def load_reader_model(model_name):
    transformer = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/'+model_name)
    return transformer, tokenizer

def answer_question(question, context, model):
    result = model(question=question, context=context)
    return result['answer']